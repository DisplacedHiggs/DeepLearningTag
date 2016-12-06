from CRABClient.UserUtilities import config
from CRABClient.UserUtilities import getUsernameFromSiteDB
from datetime import datetime
import sys

# Select dataset to crab over
number = 0

# List of datasets
datasetnames = [
"/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring16reHLT80-PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14_ext1-v1/AODSIM",
"/ZH_HToSSTobbbb_ZToLL_MH-125_MS-15_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring16DR80-premix_withHLT_80X_mcRun2_asymptotic_v14-v1/AODSIM",
"/ZH_HToSSTobbbb_ZToLL_MH-125_MS-40_ctauS-10000_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring16DR80-premix_withHLT_80X_mcRun2_asymptotic_v14-v1/AODSIM",
"/ZH_HToSSTobbbb_ZToLL_MH-125_MS-40_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring16DR80-premix_withHLT_80X_mcRun2_asymptotic_v14-v1/AODSIM",
"/ZH_HToSSTobbbb_ZToLL_MH-125_MS-40_ctauS-10_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring16DR80-premix_withHLT_80X_mcRun2_asymptotic_v14-v1/AODSIM",
"/ZH_HToSSTobbbb_ZToLL_MH-125_MS-55_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring16DR80-premix_withHLT_80X_mcRun2_asymptotic_v14-v1/AODSIM",
]
# Storage path for output files
storagepath = '/store/user/'+getUsernameFromSiteDB()+'/mwalker/NTUPLES/2016/DisplacedJet'
#storagepath = '/store/group/lpcmbja/'+getUsernameFromSiteDB()+'/2016/DisplacedDijet'

# cmsRun file
psetname = 'runDisplacedMC_wTrack_cfg.py'

# Output filename
OutputFilename = 'results.root'

# Storage site of output files
storageSite = 'T3_US_Rutgers'
#storageSite = 'T3_US_FNALLPC'

# White list sites
whiteList = ['']

# Black list sites
blackList = ['']


########## No modifications below this line are necessary ##########

dataset = filter(None, datasetnames[number].split('/'))

config = config()

#config.General.workArea = "job_"+datetime.now().strftime("%Y%m%d_%H%M%S")
#config.General.requestName = dataset[0]+"_"+dataset[1]
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = psetname 
config.JobType.outputFiles = [OutputFilename]
config.JobType.pyCfgParams = ['outputFile='+OutputFilename]

config.Data.inputDataset = datasetnames[number]
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.ignoreLocality = True
config.Data.outLFNDirBase = storagepath
config.Data.publication = False

config.Site.storageSite = storageSite

if not whiteList:
  config.Site.whitelist = whiteList

if not blackList:
  config.Site.blacklist = blackList

if __name__ == '__main__':

  from CRABAPI.RawCommand import crabCommand
  
  for dataset in datasetnames:
    print dataset
    ds = filter(None, dataset.split('/'))
    config.Data.inputDataset = dataset
    config.Data.unitsPerJob = 1
    config.Data.inputDBS = 'global'
    config.General.requestName = ds[0]
    config.General.workArea = "job_"+datetime.now().strftime("%Y%m%d")+"_"+ds[0]
    config.Data.outputDatasetTag = ds[1]+'_'+ds[2]
    
    crabCommand('submit', config = config)
