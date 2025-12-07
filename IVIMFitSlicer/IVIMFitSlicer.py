import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np
import time
import sys
import inspect
from concurrent.futures import ThreadPoolExecutor

def install_dependencies():
    needs_restart = False
    
    
    try:
        import pydicom
    except ImportError:
        slicer.util.showStatusMessage("Installing pydicom...")
        slicer.util.pip_install("pydicom")
        needs_restart = True

    
    try:
        import ivimfit
    except ImportError:
        slicer.util.showStatusMessage("Installing ivimfit library...")
        
        slicer.util.pip_install("ivimfit") 
        needs_restart = True
        
    
    try:
        import pymc
    except ImportError:
       
        slicer.util.showStatusMessage("Installing ivimfit pymc...")
        slicer.util.pip_install("pymc") 
        pass 

    if needs_restart:
        slicer.util.showStatusMessage("Dependencies installed! Reloading...", 3000)


install_dependencies()


import pydicom
try:
    import ivimfit
    from ivimfit import adc, biexp, segmented, triexp
except ImportError:
    
    pass

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class IVIMFitSlicer(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "IVIMFit Slicer"
    parent.categories = ["Diffusion"]
    parent.dependencies = []
    parent.contributors = ["Atakan Isik"]
    parent.helpText = """
    
    """
    parent.acknowledgementText = "Academic Research Tool."


class IVIMFitSlicerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "IVIM Analysis Panel"
    self.layout.addWidget(parametersCollapsibleButton)
    formLayout = qt.QFormLayout(parametersCollapsibleButton)

    
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLDiffusionWeightedVolumeNode", "vtkMRMLMultiVolumeNode", "vtkMRMLSequenceNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip("Görüntüyü Seçin")
    formLayout.addRow("Input Image:", self.inputSelector)

   
    self.maskSelector = slicer.qMRMLNodeComboBox()
    self.maskSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLSegmentationNode"]
    self.maskSelector.selectNodeUponCreation = True
    self.maskSelector.noneEnabled = False 
    self.maskSelector.setMRMLScene(slicer.mrmlScene)
    formLayout.addRow("ROI Mask:", self.maskSelector)

    
    self.bValuesLayout = qt.QVBoxLayout()
    self.bValsInput = qt.QLineEdit()
    self.bValsInput.setPlaceholderText("0, 50, ...")
    self.bValuesLayout.addWidget(self.bValsInput)
    
    self.presetBtn = qt.QPushButton("📋 Load Preset (0-900)")
    self.presetBtn.setStyleSheet("background-color: #228B22; color: white; font-weight: bold;")
    self.presetBtn.connect('clicked(bool)', self.applyPreset)
    self.bValuesLayout.addWidget(self.presetBtn)
    
    formLayout.addRow("B-Values:", self.bValuesLayout)

    
    self.methodSelector = qt.QComboBox()
    self.methodSelector.addItem("Bi-Exponential (Segmented)", "segmented")
    self.methodSelector.addItem("Mono-Exponential (ADC)", "adc")
    self.methodSelector.addItem("Bi-Exponential (Free)", "biexp")
    self.methodSelector.addItem("Tri-Exponential (Free)", "triexp")
    try:
        import ivimfit.bayesian
        self.methodSelector.addItem("Bayesian (MCMC)", "bayesian")
    except: pass
    formLayout.addRow("Algorithm:", self.methodSelector)

    
    self.scalingCheck = qt.QCheckBox("Auto-Scale Results")
    self.scalingCheck.setChecked(True)
    formLayout.addRow("Visualization:", self.scalingCheck)

    
    self.splitBFrame = qt.QWidget()
    splitLayout = qt.QHBoxLayout(self.splitBFrame)
    splitLayout.setContentsMargins(0,0,0,0)
    self.splitBSpin = qt.QSpinBox()
    self.splitBSpin.setRange(0, 3000)
    self.splitBSpin.setValue(200)
    self.splitBSpin.setSuffix(" s/mm²")
    splitLayout.addWidget(qt.QLabel("Split Threshold:"))
    splitLayout.addWidget(self.splitBSpin)
    formLayout.addRow(self.splitBFrame)

    
    self.applyButton = qt.QPushButton("🚀 Run Analysis & Plot")
    self.applyButton.enabled = False 
    self.applyButton.setStyleSheet("font-weight: bold; height: 45px; background-color: #0050aa; color: white; font-size: 14px;")
    formLayout.addRow(self.applyButton)

    self.progressBar = qt.QProgressBar()
    self.progressBar.visible = False
    self.layout.addWidget(self.progressBar)
    
    
    self.resultsGroup = ctk.ctkCollapsibleButton()
    self.resultsGroup.text = "ROI Averaged Results"
    self.resultsGroup.collapsed = False
    self.layout.addWidget(self.resultsGroup)
    
    self.resultsLayout = qt.QFormLayout(self.resultsGroup)
    self.resultLabels = {}
    params = ["ADC", "D", "D_star", "f", "D_slow_tri", "D_inter_tri", "D_fast_tri", "f_fast", "f_inter", "f_slow"]
    for p in params:
        lbl = qt.QLineEdit()
        lbl.setReadOnly(True)
        lbl.setStyleSheet("font-weight: bold; color: #333;")
        self.resultsLayout.addRow(f"{p}:", lbl)
        self.resultLabels[p] = lbl
        self.resultsLayout.labelForField(lbl).setVisible(False)
        lbl.setVisible(False)

    self.layout.addStretch(1)

   
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtonState)
    self.maskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtonState)
    self.methodSelector.connect("currentIndexChanged(int)", self.onMethodChanged)
    self.bValsInput.textChanged.connect(self.updateButtonState)

    self.onMethodChanged()
    self.updateButtonState()

  def onMethodChanged(self):
    self.splitBFrame.visible = ("segmented" in self.methodSelector.currentData)
    self.updateResultVisibility(self.methodSelector.currentData)

  def updateResultVisibility(self, method):
    for p, lbl in self.resultLabels.items():
        lbl.setVisible(False)
        self.resultsLayout.labelForField(lbl).setVisible(False)
        lbl.setText("")
    
    to_show = []
    if method == "adc": to_show = ["ADC"]
    elif method in ["segmented", "biexp", "bayesian"]: to_show = ["f", "D", "D_star"]
    elif method == "triexp": to_show = ["f_fast", "f_inter", "f_slow", "D_slow_tri", "D_inter_tri", "D_fast_tri"]
        
    for p in to_show:
        if p in self.resultLabels:
            self.resultLabels[p].setVisible(True)
            self.resultsLayout.labelForField(self.resultLabels[p]).setVisible(True)

  def applyPreset(self):
    self.bValsInput.setText("0, 50, 100, 150, 200, 300, 400, 500, 700, 900")
    self.updateButtonState()

  def updateButtonState(self):
    self.applyButton.enabled = (self.inputSelector.currentNode() is not None) and \
                               (self.maskSelector.currentNode() is not None) and \
                               (len(self.bValsInput.text) > 0)

  def onApplyButton(self):
    try:
        b_vals = [float(x.strip()) for x in self.bValsInput.text.split(',')]
        if len(b_vals) < 2: raise ValueError
    except:
        slicer.util.errorDisplay("B-değerleri hatalı.")
        return

    maskNode = self.maskSelector.currentNode()
    if not maskNode: return

    self.logic = IVIMFitSlicerLogic()
    self.applyButton.enabled = False
    self.progressBar.visible = True
    self.progressBar.setRange(0, 0)
    slicer.app.processEvents()

    try:
        vol = self.inputSelector.currentNode()
        method = self.methodSelector.currentData
        split = self.splitBSpin.value
        use_scaling = self.scalingCheck.isChecked()
        
        
        results_package = self.logic.process(vol, maskNode, b_vals, method, split, use_scaling, self.progressBar)
        
        
        self.displayResults(results_package['params'], method)
        
        
        self.plotResults(results_package)
        
        slicer.util.showStatusMessage("Analysis Complete!", 3000)
        
    except Exception as e:
        slicer.util.errorDisplay(f"Error:\n{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        self.applyButton.enabled = True
        self.progressBar.visible = False
        self.updateButtonState()

  def displayResults(self, params, method):
    if method == "adc":
        self.resultLabels['ADC'].setText(f"{params.get('D',0):.6f}mm^2/s")
    elif method in ["segmented", "biexp", "bayesian"]:
        self.resultLabels['f'].setText(f"{params.get('f',0)*100:.2f} %")
        self.resultLabels['D'].setText(f"{params.get('D',0):.6f}mm^2/s")
        self.resultLabels['D_star'].setText(f"{params.get('Ds',0):.6f}mm^2/s")
    elif method == "triexp":
        self.resultLabels['f_fast'].setText(f"{params.get('f',0)*100:.2f} %")
        self.resultLabels['f_inter'].setText(f"{params.get('f2',0)*100:.2f} %")
       
        f_slow = 1.0 - (params.get('f',0)+params.get('f2',0))
        self.resultLabels['f_slow'].setText(f"{f_slow*100:.2f} %")
        self.resultLabels['D_slow_tri'].setText(f"{params.get('D',0):.6f}mm^2/s")
        self.resultLabels['D_fast_tri'].setText(f"{params.get('Ds',0):.6f}mm^2/s")
        self.resultLabels['D_inter_tri'].setText(f"{params.get('Ds2',0):.6f}mm^2/s")

  def plotResults(self, data):
    b_values = data['b']
    signal_avg = data['signal_avg']
    signal_fit = data['signal_fit']
    r2 = data['r2']
    params = data['params']
    method = data['method']
    
    
    if method in ["segmented", "biexp", "bayesian"]:
        legend_txt = f"Fit: f={params.get('f',0):.2f}, D={params.get('D',0):.6f}, D*={params.get('Ds',0):.6f}"
    elif method == "adc":
        legend_txt = f"Fit: ADC={params.get('D',0):.6f}"
    elif method == "triexp":
        f1 = params.get('f', 0); f2 = params.get('f2', 0); f3 = 1.0 - (f1+f2)
        D = params.get('D', 0); Ds1 = params.get('Ds', 0); Ds2 = params.get('Ds2', 0)
        legend_txt = f"Fit: f1={f1:.2f}, f2={f2:.2f}, f3={f3:.2f} | D={D:.6f}, D*1={Ds1:.6f}, D*2={Ds2:.6f}"
    else: legend_txt = "Model Fit"

    
    tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "IVIM_ROI_Data")
    tableNode.RemoveAllColumns()
    
    arr_b = vtk.vtkDoubleArray(); arr_b.SetName("B-Value")
    arr_raw = vtk.vtkDoubleArray(); arr_raw.SetName("Measured Signal")
    arr_fit = vtk.vtkDoubleArray(); arr_fit.SetName("Fit Curve")
    
    
    sorted_idx = np.argsort(b_values)
    for i in sorted_idx:
        arr_b.InsertNextValue(float(b_values[i])) 
        arr_raw.InsertNextValue(float(signal_avg[i]))
        arr_fit.InsertNextValue(float(signal_fit[i]))
    
    tableNode.AddColumn(arr_b); tableNode.AddColumn(arr_raw); tableNode.AddColumn(arr_fit)
    
    
    chartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "IVIM_Chart")
    chartNode.SetTitle(f"IVIM Curve (R²={r2:.4f})")
    chartNode.SetXAxisTitle("b-value (s/mm²)")
    chartNode.SetYAxisTitle("Normalized Signal (S/S0)")
    chartNode.SetLegendVisibility(True)
    
    
    s1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Raw Data")
    s1.SetName("Measured Data")
    s1.SetAndObserveTableNodeID(tableNode.GetID())
    s1.SetXColumnName("B-Value")
    s1.SetYColumnName("Measured Signal")
    s1.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter) 
    s1.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleDashDot) 
    s1.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCross) 
    s1.SetColor(0, 0, 0) # Siyah
    s1.SetMarkerSize(8)
    chartNode.AddAndObservePlotSeriesNodeID(s1.GetID())
    
    
    s2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Fit Data")
    s2.SetName(legend_txt) 
    s2.SetAndObserveTableNodeID(tableNode.GetID())
    s2.SetXColumnName("B-Value")
    s2.SetYColumnName("Fit Curve")
    s2.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter) 
    s2.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid) 
    s2.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone) 
    s2.SetColor(0.8, 0, 0) 
    s2.SetLineWidth(3)
    chartNode.AddAndObservePlotSeriesNodeID(s2.GetID())
    
    
    lm = slicer.app.layoutManager()
    lm.setLayout(24) 
    pw = lm.plotWidget(0)
    if pw: 
        pv = pw.mrmlPlotViewNode()
        pv.SetPlotChartNodeID(chartNode.GetID())
        pv.SetInteractionMode(slicer.vtkMRMLPlotViewNode.InteractionModePanView)


class IVIMFitSlicerLogic(ScriptedLoadableModuleLogic):
  
  def process(self, inputNode, maskNode, b_values, method, split_b, use_scaling, progressBar):
    startTime = time.time()
    
    
    input_array, refNode = self.extract_pixel_data_safe(inputNode, len(b_values))
    Z, Y, X, B = input_array.shape
    
    if B != len(b_values):
        if B == len(b_values) + 1: b_values.insert(0, 0.0)
        else: raise ValueError(f"Frame ({B}) != B-Values ({len(b_values)})")

    
    mask_array = self.prepare_mask(maskNode, inputNode)
    if mask_array.shape != (Z,Y,X): pass
    indices = np.where(mask_array > 0)
    n_pixels = len(indices[0])
    if n_pixels == 0: raise ValueError("Mask is empty!")

    slicer.app.processEvents()

    
    map_method = "segmented" if method == "bayesian" else method
    signals = input_array[indices]
    map_results = np.zeros((n_pixels, 5), dtype=np.float32)
    b_vals_arr = np.array(b_values, dtype=np.float64)
    
    self.run_fitting(b_vals_arr, signals, map_method, split_b, map_results, progressBar)

    
    map_scaled = map_results.copy()
    if use_scaling:
        if map_method == "adc": map_scaled[:, 0] *= 1e6
        elif map_method in ["segmented", "biexp", "bayesian"]:
            map_scaled[:, 0] *= 100; map_scaled[:, 1] *= 1e6; map_scaled[:, 2] *= 1e6
        elif map_method == "triexp":
            map_scaled[:, 0] *= 100; map_scaled[:, 1] *= 100
            map_scaled[:, 2] *= 1e6; map_scaled[:, 3] *= 1e6; map_scaled[:, 4] *= 1e6

    out_maps = {k: np.zeros((Z,Y,X), dtype=np.float32) for k in ['f','D','Ds','f2','Ds2', 'f_slow']}
    if map_method == "adc": out_maps['D'][indices] = map_scaled[:, 0]
    elif map_method in ["segmented", "biexp", "bayesian"]:
        out_maps['f'][indices] = map_scaled[:, 0]; out_maps['D'][indices] = map_scaled[:, 1]; out_maps['Ds'][indices] = map_scaled[:, 2]
    elif map_method == "triexp":
        out_maps['f'][indices] = map_scaled[:, 0]; out_maps['f2'][indices] = map_scaled[:, 1]
        out_maps['D'][indices] = map_scaled[:, 2]; out_maps['Ds'][indices] = map_scaled[:, 3]; out_maps['Ds2'][indices] = map_scaled[:, 4]
        out_maps['f_slow'][indices] = np.clip(100.0 - (map_scaled[:, 0] + map_scaled[:, 1]), 0, 100)

    bn = inputNode.GetName()
    final_D_node = self.save_volume(out_maps['D'], refNode, f"{bn}_{method}_D_Map")
    self.force_show_volume(final_D_node)

    if map_method != "adc":
        self.save_volume(out_maps['f'], refNode, f"{bn}_{method}_f_Map")
        self.save_volume(out_maps['Ds'], refNode, f"{bn}_{method}_Ds_Map")
    if map_method == "triexp":
        self.save_volume(out_maps['f2'], refNode, f"{bn}_{method}_f_Interm")
        self.save_volume(out_maps['Ds2'], refNode, f"{bn}_{method}_Ds_Interm")
        self.save_volume(out_maps['f_slow'], refNode, f"{bn}_{method}_f_Slow")

    
    avg_signal = np.mean(signals, axis=0)
    s0 = avg_signal[0] if avg_signal[0] != 0 else 1.0
    avg_signal_norm = avg_signal / s0
    
    roi_params = {}
    fit_curve = np.zeros_like(avg_signal_norm)
    
    from ivimfit import adc, biexp, segmented, triexp
    try: from ivimfit import bayesian
    except: pass

    try:
        if method == "segmented":
            p = segmented.fit_biexp_segmented(b_vals_arr, avg_signal_norm, split_b=split_b)
            roi_params = {'f': p[0], 'D': p[1], 'Ds': p[2]}
            fit_curve = biexp.biexp_model(b_vals_arr, p[0], p[1], p[2])
        elif method == "adc":
            d = adc.fit_adc(b_vals_arr, avg_signal_norm)
            roi_params = {'D': d}
            fit_curve = adc.monoexp_model(b_vals_arr, d)
        elif method == "biexp":
            p = biexp.fit_biexp_free(b_vals_arr, avg_signal_norm)
            roi_params = {'f': p[0], 'D': p[1], 'Ds': p[2]}
            fit_curve = biexp.biexp_model(b_vals_arr, p[0], p[1], p[2])
        elif method == "triexp":
            p = triexp.fit_triexp_free(b_vals_arr, avg_signal_norm)
            roi_params = {'f': p[0], 'f2': p[1], 'D': p[2], 'Ds': p[3], 'Ds2': p[4]}
            fit_curve = triexp.triexp_model(b_vals_arr, p[0], p[1], p[2], p[3], p[4])
        elif method == "bayesian" and 'bayesian' in locals():
            p = bayesian.fit_bayesian(b_vals_arr, avg_signal_norm, draws=2000, chains=1, progressbar=False)
            roi_params = {'f': p[0], 'D': p[1], 'Ds': p[2]}
            fit_curve = biexp.biexp_model(b_vals_arr, p[0], p[1], p[2])
    except: pass

    ss_res = np.sum((avg_signal_norm - fit_curve)**2)
    ss_tot = np.sum((avg_signal_norm - np.mean(avg_signal_norm))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'b': b_values, 'signal_avg': avg_signal_norm, 'signal_fit': fit_curve,
        'params': roi_params, 'r2': r2, 'method': method
    }

  def force_show_volume(self, volumeNode):
    slicer.util.setSliceViewerLayers(background=volumeNode)
    slicer.util.resetSliceViews()

  def extract_pixel_data_safe(self, node, expected_b):
    if node.IsA("vtkMRMLSequenceNode"):
        n = node.GetNumberOfDataNodes(); lst = []; ref = node.GetDataNodeAtValue(node.GetNthIndexValue(0))
        for i in range(n): lst.append(slicer.util.arrayFromVolume(node.GetDataNodeAtValue(node.GetNthIndexValue(i))).copy())
        return np.moveaxis(np.array(lst), 0, -1), ref
    try:
        bn = slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(node)
        if bn:
            sn = bn.GetMasterSequenceNode(); n = sn.GetNumberOfDataNodes(); lst = []; ref = node; orig = bn.GetSelectedItemNumber()
            for i in range(n): bn.SetSelectedItemNumber(i); slicer.app.processEvents(); lst.append(slicer.util.arrayFromVolume(node).copy())
            bn.SetSelectedItemNumber(orig); return np.moveaxis(np.array(lst), 0, -1), ref
    except: pass
    arr = slicer.util.arrayFromVolume(node)
    if arr.ndim == 4:
        if arr.shape[0] == expected_b or arr.shape[0] == expected_b+1: return np.moveaxis(arr, 0, -1), node
        if arr.shape[-1] == expected_b or arr.shape[-1] == expected_b+1: return arr, node
    return arr, node

  def prepare_mask(self, maskNode, inputNode):
    if maskNode.IsA("vtkMRMLSegmentationNode"):
        ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(maskNode, ln, inputNode)
        arr = slicer.util.arrayFromVolume(ln); slicer.mrmlScene.RemoveNode(ln)
        return arr[0] if arr.ndim==4 else arr
    arr = slicer.util.arrayFromVolume(maskNode)
    return arr[0] if arr.ndim==4 else arr

  def save_volume(self, data, ref, name):
    v = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, ref, name)
    slicer.util.updateVolumeFromArray(v, data)
    v.SetAttribute("Quantities", "Scalar")
    d = v.GetDisplayNode()
    if d and data.max()>0:
        d.AutoWindowLevelOff(); flat = data.flatten(); vld = flat[flat>0]
        if vld.size>0: p1=np.percentile(vld,1); p99=np.percentile(vld,99); d.SetWindowLevel(p99-p1, p1+(p99-p1)/2)
    return v

  def run_fitting(self, b_vals, signals, method, split_b, results, progressBar):
    from ivimfit import adc, biexp, segmented, triexp
    n_total = len(signals); chunk_size = 500; n_chunks = int(np.ceil(n_total / chunk_size))
    if progressBar: progressBar.setRange(0, n_chunks)
    def worker(start_idx):
        end = min(start_idx + chunk_size, n_total); chunk = signals[start_idx:end]; res = np.zeros((len(chunk), 5), dtype=np.float32)
        for i, sig in enumerate(chunk):
            try:
                if method in ["segmented", "bayesian"]: r = segmented.fit_biexp_segmented(b_vals, sig, split_b=split_b); res[i, :3] = r
                elif method == "adc": res[i, 0] = adc.fit_adc(b_vals, sig)
                elif method == "biexp": r = biexp.fit_biexp_free(b_vals, sig); res[i, :3] = r
                elif method == "triexp": r = triexp.fit_triexp_free(b_vals, sig); res[i] = r
            except: pass
        return start_idx, res
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, i*chunk_size) for i in range(n_chunks)]
        for i, f in enumerate(futures):
            idx, r = f.result(); results[idx:idx+len(r)] = r

            if progressBar: progressBar.setValue(i+1); slicer.app.processEvents()
