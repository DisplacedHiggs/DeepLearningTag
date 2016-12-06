#include "helperDisplacedDijets.C"
#include "helperDisplacedDijetTriggers.C"
#include "helperCNN.C"

void makeCNNOutput(const char* ifname="./sig_wtracks.root"
		  , const char* ofname="exampleAnalyzed.root"
		  , bool isSignal = false
		  , int mode = 0
		       , bool isMC = true
		  , int maxEvents = 1000000
		  , const char* json = "/home/mwalker/golden_246908-260627_20151120.txt"
)
{
  TChain* tree = new TChain("tree");
  TString input=ifname;
  //input += "/*.root";
  tree->Add(input);

  FlatTreeReader* reader = new FlatTreeReader(tree);

  AdvancedHandler* handler = new AdvancedHandler(ofname,reader);

  AnalysisTreeWriter* writer = new AnalysisTreeWriter(handler,"treeR");

  handler->setWriter(writer);

  ////////////////////////////////////////////////////
  //Setup some products jets, electrons, muons, etc//
  ///////////////////////////////////////////////////
  setupProducts(handler,isSignal);
  setupTracks(handler);
  setupTrackMatching(handler);
  //isSignal->addProductCut("goodSignalJets","fromSecondary");

  //////////////////////
  //Add some variables//
  //////////////////////
  vector<TString> products = {"BASICCALOJETS","BASICCALOJETS1"};

  setupVariables(handler);
  setupListVariablesAndHistograms(handler,products);
  setupTriggers(handler);
  //setupListMaxVariablesAndHistograms(handler);
  setupMC(handler);

  setupCNNVariables(handler);
  //else handler->readGoodRunLumiFromJSON(TString(json));

  ////////////////////////
  //Add some signatures//
  //////////////////////
  //handler->addSignature("SigMET150","")
  //  ->addCut("MET150")
  //  ;

  //////////////////////////
  //Create some histograms//
  //////////////////////////
  //addHistograms(handler);

  //////////////////////////////////
  //Final bookkeeping and execution//
  ///////////////////////////////////

  handler->setMode("nEntryHigh",maxEvents);
  //handler->setMode("nEntryHigh",1);
  //handler->setDebugMode(true);
  //handler->addPrintModule(new PrintModuleEverything("everything"));

  handler->initSignatures();
  handler->eventLoop();
  //handler->eventLoop(1,24248);
  handler->finishSignatures();

  cout<<"Done, exiting ...."<<endl;
}
