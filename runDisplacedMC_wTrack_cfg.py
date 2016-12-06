import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.StandardSequences.Eras import eras
process = cms.Process('RutgersAOD',eras.Run2_25ns)#for 25ns 13 TeV data
#process = cms.Process("RutgersAOD")
options = VarParsing.VarParsing ('analysis')

#set default arguments
#options.inputFiles='root://cmsxrootd.fnal.gov//store/data/Run2015D/JetHT/AOD/16Dec2015-v1/00000/0A2C6696-AEAF-E511-8551-0026189438EB.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/data/Run2016G/SingleElectron/AOD/PromptReco-v1/000/278/820/00000/9A8A2744-5064-E611-A3D2-FA163EF92E66.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/data/Run2016G/SingleMuon/AOD/PromptReco-v1/000/278/820/00000/00107902-2364-E611-AF03-02163E0141B8.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/data/Run2016G/SingleMuon/AOD/PromptReco-v1/000/278/820/00000/02B2A5DA-2E64-E611-BFB4-02163E0144F8.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/data/Run2016G/SingleMuon/AOD/PromptReco-v1/000/278/820/00000/02D8AEAC-1E64-E611-B55E-02163E0135F9.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/mc/RunIISpring16DR80/WplusH_HToSSTodddd_WToLNu_MH-125_MS-7_ctauS-10_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/premix_withHLT_80X_mcRun2_asymptotic_v14-v1/10000/3C3EA3EE-026D-E611-AD6D-0CC47AA989BA.root'
#options.inputFiles='root://cmsxrootd.fnal.gov//store/mc/RunIISpring16DR80/ZH_HToSSTobbbb_ZToLL_MH-125_MS-40_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/premix_withHLT_80X_mcRun2_asymptotic_v14-v1/90000/1ABF07E2-CE6C-E611-A80E-20CF3019DF03.root'
options.inputFiles='root://cmsxrootd.fnal.gov//store/mc/RunIISpring16DR80/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/AODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/041D3A34-1000-E611-A66E-0CC47A4C8E16.root'
options.outputFile='test.root'
#options.inputFiles= '/store/relval/CMSSW_7_0_0/RelValProdTTbar_13/AODSIM/POSTLS170_V3-v2/00000/40D11F5C-EA98-E311-BE17-02163E00E964.root'
#options.inputFiles= 'file:/cms/thomassen/2012/Signal/StopRPV/store/aodsim/LLE122/StopRPV_8TeV_chunk3_stop950_bino800_LLE122_aodsim.root'
#options.maxEvents = 100 # -1 means all even
options.maxEvents = -1

# get and parse the command line arguments
options.parseArguments()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
)

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("TrackingTools.MaterialEffects.MaterialPropagator_cfi")
process.load('PhysicsTools.PatAlgos.patSequences_cff')

process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi")

#process.load("RutgersAODReader.BaseAODReader.displacedAOD_cfi")
#process.load("DisplacedDijet.DisplacedJetAnlzr.DJ_DiJetVertices_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
)
process.options.allowUnscheduled = cms.untracked.bool( True )

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
dataFormat = DataFormat.AOD
switchOnVIDPhotonIdProducer(process, dataFormat)
my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff']
#['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff',
#                 'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff']
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection) 
#
# ELECTRON IDs
switchOnVIDElectronIdProducer(process, dataFormat)
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Switch to calibrated electrons in switchOnVIDElectronIdProducer (with energy smearing and scale corrections):
#   https://github.com/cms-sw/cmssw/blob/CMSSW_7_6_X/PhysicsTools/SelectorUtils/python/tools/vid_id_tools.py
#   https://github.com/cms-sw/cmssw/blob/CMSSW_7_6_X/RecoEgamma/ElectronIdentification/python/egmGsfElectronIDs_cff.py
#       egmGsfElectronIDSequence = cms.Sequence( electronMVAValueMapProducer * egmGsfElectronIDs * electronRegressionValueMapProducer)
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# For HEEP ID please see slide 13 here: https://indico.cern.ch/event/522084/contributions/2138101/attachments/1260278/1862267/ZPrime2016Status.pdf
# - HEEP ID broke in 76X
# - Temporary fix for 76X in (HEEP V6.1)
# - Permanent fix in for 81X
# - Working on a miniAODfix for 80X --- This doesnt seem to be available as of May-10-2016 (https://twiki.cern.ch/twiki/bin/view/CMS/HEEPElectronIdentificationRun2)
# -------------------------------------------------------------------------------------------------------------------------------------------------
# customization -begin : needed to use "calibratedPatElectrons" in the output.
#process.egmGsfElectronIDs.physicsObjectIDs = cms.VPSet()                                           # -- DISABLED
#process.egmGsfElectronIDs.physicsObjectSrc = cms.InputTag("calibratedPatElectrons")                # -- DISABLED
#dataFormatString = "MiniAOD"                                                                       # -- DISABLED
#process.electronMVAValueMapProducer.srcMiniAOD =  cms.InputTag("calibratedPatElectrons","")        # -- DISABLED
#process.electronRegressionValueMapProducer.srcMiniAOD =  cms.InputTag("calibratedPatElectrons","") # -- DISABLED
# customization -end
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff',
                 'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff',
                 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff',
                 'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff']
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)


from RutgersAODReader.BaseAODReader.displacedAOD_cfi import displacedAOD

process.displacedAOD = displacedAOD.clone()

process.displacedAOD.outFilename=options.outputFile
#process.displacedAOD.patPhotonsInputTag = "gedPhotons"
process.displacedAOD.JECUncFileName = cms.FileInPath("RutgersAODReader/BaseAODReader/data/Fall15_25nsV2_DATA_Uncertainty_AK4PFchs.txt")
process.displacedAOD.JERSFfileName = cms.FileInPath("RutgersAODReader/BaseAODReader/data/Summer15_25nsV6_MC_SF_AK4PFchs.txt")
process.displacedAOD.processTracks = cms.bool(True)
process.displacedAOD.setupList = cms.VPSet(
    cms.PSet(type=cms.untracked.string("ObjectVariableMethod"),
             name=cms.string("PT"),
             methodName=cms.string("Pt"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValueInList"),
             name=cms.string("SaveAllOfType"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             values=cms.vstring("photon","vertex","beamspot","mc","met","electron","muon","trigger","filter","hcalnoise","kshort","track"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableInRange"),
             name=cms.string("PT5"),
             variableName=cms.string("PT"),
             variableType=cms.string("double"),
             low=cms.double(5),
             high=cms.double(1000000),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableInRange"),
             name=cms.string("PT10"),
             variableName=cms.string("PT"),
             variableType=cms.string("double"),
             low=cms.double(10),
             high=cms.double(1000000),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableInRange"),
             name=cms.string("PT15"),
             variableName=cms.string("PT"),
             variableType=cms.string("double"),
             low=cms.double(15),
             high=cms.double(1000000),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValueInList"),
             name=cms.string("isTrack"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             values=cms.vstring("losttrack","pftrack","track"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("SaveTrack"),
             cutList=cms.vstring("isTrack","PT5"),
             doAnd=cms.bool(True),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValue"),
             name=cms.string("isMuon"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             value=cms.string("muon"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValue"),
             name=cms.string("isElectron"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             value=cms.string("electron"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValueInList"),
             name=cms.string("isJet"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             values=cms.vstring("jet","calojet"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableValue"),
             name=cms.string("isTau"),
             variableName=cms.string("INPUTTYPE"),
             variableType=cms.string("TString"),
             value=cms.string("tau"),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("SaveJet"),
             cutList=cms.vstring("isJet","PT5"),
             doAnd=cms.bool(True),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("SaveTau"),
             cutList=cms.vstring("isTau","PT15"),
             doAnd=cms.bool(True),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("SaveMuon"),
             cutList=cms.vstring("isMuon","PT5"),
             doAnd=cms.bool(True),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("SaveElectron"),
             cutList=cms.vstring("isElectron","PT5"),
             doAnd=cms.bool(True),
             ),
    cms.PSet(type=cms.untracked.string("ObjectVariableCombined"),
             name=cms.string("WRITEOBJECT"),
             cutList=cms.vstring("SaveJet","SaveTrack","SaveAllOfType","SaveMuon","SaveElectron","SaveTau"),
             doAnd=cms.bool(False),
             ),
    cms.PSet(type=cms.untracked.string("Signature"),
             name=cms.string("testSignature"),
             cutList=cms.vstring(),
             ),
)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')

# for data:
from PhysicsTools.PatAlgos.tools.coreTools import runOnData
runOnData( process ,names=['Photons', 'Electrons','Muons', 'Taus', 'METs', 'PFAll', 'PFElectrons','PFTaus','PFMuons'] )
#runOnData( process ,outputModules = [])

process.p = cms.Path(
    process.egmPhotonIDSequence *
    process.displacedAOD)
