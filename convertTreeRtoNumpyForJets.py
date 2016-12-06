from ROOT import *
import numpy as np
import sys
import math

rCone = 0.4
nBins = 8
gROOT.SetBatch(True)

def getIbin(deta):
    ibin = 0
    if deta > rCone:
        ibin = nBins - 1
    elif deta < -rCone:
        ibin = 0
    else:
        ibin = int((deta + rCone) / (2*rCone) * nBins)
    return ibin


def main():
    if len(sys.argv) != 3:
        print "usage: python convertTreeRtoNumpyForJets.py inputfilename outputfilename"
        return

    ifname = sys.argv[1]
    ofname = sys.argv[2]

    maxToSave = 40000

    infile = TFile(ifname)
    if not infile.IsOpen():
        print "file not open / doesn't exist"
        return

    tree = infile.Get("treeR")
    jets = []
    for i in range(0,tree.GetEntries()):
        #if i < 40000: continue
        if len(jets) > maxToSave: 
            print "stopping on event:",i
            break
        tree.GetEntry(i)
        #if len(tree.TRACKSMATCHED_WHICHJET) != len(tree.TRACKSMATCHED_LOG10TRACKANGLE): continue
        if len(tree.MISSINGINNER_BASICCALOJETS1MATCHED) != len(tree.MEDIANLOG10TRACKANGLE_BASICCALOJETS1MATCHED): continue

        for j in range(0,len(tree.MISSINGINNER_BASICCALOJETS1MATCHED)):
            #print tree.JETINDEX_BASICCALOJETS[j]
            jet = np.zeros(shape=(17))
            jet[0] = tree.PT_BASICCALOJETS1MATCHED[j]
            jet[1] = tree.ETA_BASICCALOJETS1MATCHED[j]
            jet[2] = tree.PHI_BASICCALOJETS1MATCHED[j]
            jet[3] = tree.MISSINGINNER_BASICCALOJETS1MATCHED[j]
            jet[4] = tree.MISSINGOUTER_BASICCALOJETS1MATCHED[j]
            jet[5] = tree.ALPHAMAX_BASICCALOJETS1MATCHED[j]
            jet[6] = tree.ASSOCAPLANARITY_BASICCALOJETS1MATCHED[j] if not math.isnan(tree.ASSOCAPLANARITY_BASICCALOJETS1MATCHED[j]) else -1
            jet[7] = tree.ASSOCSPHERICITY_BASICCALOJETS1MATCHED[j] if not math.isnan(tree.ASSOCSPHERICITY_BASICCALOJETS1MATCHED[j]) else -1
            jet[8] = tree.ASSOCTHRUSTMAJOR_BASICCALOJETS1MATCHED[j]
            jet[9] = tree.ASSOCTHRUSTMINOR_BASICCALOJETS1MATCHED[j]
            jet[10] = tree.BETA_BASICCALOJETS1MATCHED[j]
            #jet[11] = tree.LEPANGLE_DPHI_BASICCALOJETS1MATCHED[j]
            jet[11] = tree.LINEARRADIALMOMENT_BASICCALOJETS1MATCHED[j]
            jet[12] = tree.MEDIANIPLOG10SIG_BASICCALOJETS1MATCHED[j]
            jet[13] = tree.MEDIANLOG10TRACKANGLE_BASICCALOJETS1MATCHED[j]
            jet[14] = tree.METANGLE_DPHI_BASICCALOJETS1MATCHED[j]
            jet[15] = tree.SUMIPSIG_BASICCALOJETS1MATCHED[j]
            jet[16] = tree.TOTALTRACKPT_BASICCALOJETS1MATCHED[j]
            #print jet
            jets.append(jet)


    jetarray = np.zeros(shape=(len(jets),17))
    for i,j in enumerate(jets):
        jetarray[i] = j
    print "collected:",len(jets),"jets"
    np.save(ofname,jetarray)
    return
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
