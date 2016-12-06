void setupTracks(AdvancedHandler* handler)
{

}

void setupTrackMatching(AdvancedHandler* handler)
{

  ObjectAssociationVariableMatch<int>* objMatch = new ObjectAssociationVariableMatch<int>("whichJet","jetIndex","JETMATCH");

  handler->addProduct("TRACKSMATCHEDTEMP","ALLTRACKS");
  handler->addProductAssociation("TRACKSMATCHEDTEMP","BASICCALOJETSMATCHED",objMatch,true);

  ObjectAssociationDeltaR* deltaR = new ObjectAssociationDeltaR(100,"trackMatch");
  handler->addProductAssociation("TRACKSMATCHEDTEMP","TRACKSMATCHEDTEMP",deltaR,true);
  handler->addAssociateVariable("TRACKSMATCHEDTEMP","SELFTRACKMATCH",new ObjectVariableAssociateAngles("trackMatch","SELFTRACKMATCH"));

  handler->setDefaultObjectVariable("TRACKSMATCHEDTEMP","Log10trackAngle",-50.0,false);
  handler->setDefaultObjectVariable("TRACKSMATCHEDTEMP","trackAngle",-50.0,false);

  handler->addProduct("TRACKSMATCHED","TRACKSMATCHEDTEMP");
  handler->addProductCut("TRACKSMATCHED","hasAssociate_JETMATCH");

  handler->setDefaultObjectVariable("TRACKSMATCHED","Log10trackAngle",-50.0,false);
  handler->setDefaultObjectVariable("TRACKSMATCHED","trackAngle",-50.0,false);

}

void setupCNNVariables(AdvancedHandler* handler)
{
  handler->addEventVariable("TRACKSMATCHED_WHICHJET",new EventVariableObjectVariableVector<int>("whichJet","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_WHICHVERTEX",new EventVariableObjectVariableVector<int>("whichVertex","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_CHARGE",new EventVariableObjectVariableVector<int>("charge","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_NMISSINGINNER",new EventVariableObjectVariableVector<int>("nMissingInner","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_NMISSINGOUTER",new EventVariableObjectVariableVector<int>("nMissingOuter","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_ETA",new EventVariableObjectVariableVector<double>("ETA","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_PHI",new EventVariableObjectVariableVector<double>("PHI","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_PT",new EventVariableObjectVariableVector<double>("PT","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_LOG10SUMIPSIG",new EventVariableObjectVariableVector<double>("Log10SumIPSig","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_LOG10TRACKANGLE",new EventVariableObjectVariableVector<double>("Log10trackAngle","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_DXY",new EventVariableObjectVariableVector<double>("dxy","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_DXYSIG",new EventVariableObjectVariableVector<double>("dxySig","TRACKSMATCHED"));
  handler->addEventVariable("TRACKSMATCHED_TRACKANGLE",new EventVariableObjectVariableVector<double>("trackAngle","TRACKSMATCHED"));

  handler->addEventVariable("SELFTRACKSMATCHED_DETA", new EventVariableObjectVariableVector<double>("SELFTRACKMATCH_dETA","TRACKSMATCHED"));
  handler->addEventVariable("SELFTRACKSMATCHED_DPHI", new EventVariableObjectVariableVector<double>("SELFTRACKMATCH_dPHI","TRACKSMATCHED"));



}
