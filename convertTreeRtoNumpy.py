from ROOT import *
import numpy as np
import sys

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
        print "usage: python convertTreeRtoNumpy.py inputfilename outputfilename"
        return

    ifname = sys.argv[1]
    ofname = sys.argv[2]

    maxToSave = 20000

    infile = TFile(ifname)
    if not infile.IsOpen():
        print "file not open / doesn't exist"
        return

    tree = infile.Get("treeR")
    jets = []
    for i in range(0,tree.GetEntries()):
        tree.GetEntry(i)
        if len(tree.TRACKSMATCHED_WHICHJET) != len(tree.TRACKSMATCHED_LOG10TRACKANGLE): continue

        calojets = dict()
        imagejets = dict()
        for j in range(0,len(tree.JETINDEX_BASICCALOJETSMATCHED)):
            #print tree.JETINDEX_BASICCALOJETS[j]
            jvec = TLorentzVector()
            jvec.SetPtEtaPhiM(tree.PT_BASICCALOJETSMATCHED[j],tree.ETA_BASICCALOJETSMATCHED[j],tree.PHI_BASICCALOJETSMATCHED[j],0)
            calojets[tree.JETINDEX_BASICCALOJETSMATCHED[j]] = jvec
            jet = np.zeros(shape=(nBins,nBins,4))
            imagejets[tree.JETINDEX_BASICCALOJETSMATCHED[j]] = jet

        for j in range(0,len(tree.TRACKSMATCHED_WHICHJET)):
            whichJet = tree.TRACKSMATCHED_WHICHJET[j]
            if not calojets.has_key(whichJet): continue
            eta = tree.TRACKSMATCHED_ETA[j]
            phi = tree.TRACKSMATCHED_PHI[j]
            pt = tree.TRACKSMATCHED_PT[j]#0
            tvec = TLorentzVector()
            tvec.SetPtEtaPhiM(pt,eta,phi,0)
            deta = jvec.Eta() - eta
            dphi = jvec.DeltaPhi(tvec)
            ibin = getIbin(deta)
            jbin = getIbin(dphi)
            #print j,deta,dphi,ibin,jbin
            pt = np.log10(pt)/10.0
            #print j
            l10dxysig = tree.TRACKSMATCHED_LOG10SUMIPSIG[j]/10.0#1
            l10trackangle = tree.TRACKSMATCHED_LOG10TRACKANGLE[j]#2
            l10dxy = np.log10(tree.TRACKSMATCHED_DXY[j])/10.0#3
            charge = tree.TRACKSMATCHED_CHARGE[j]#4
            missinginner = tree.TRACKSMATCHED_NMISSINGINNER[j]#5
            missingouter = tree.TRACKSMATCHED_NMISSINGOUTER[j]#6
            whichVertex = tree.TRACKSMATCHED_WHICHVERTEX[j]#7
            
            jvec = calojets[whichJet]
            ijet = imagejets[whichJet]

            if pt > ijet[ibin][jbin][0]:
                imagejets[whichJet][ibin][jbin][0] = pt
                imagejets[whichJet][ibin][jbin][1] = l10dxysig
                imagejets[whichJet][ibin][jbin][2] = l10trackangle
                imagejets[whichJet][ibin][jbin][3] = missinginner
                #imagejets[whichJet][ibin][jbin][3] = l10dxy
                #imagejets[whichJet][ibin][jbin][4] = charge
                #imagejets[whichJet][ibin][jbin][5] = missinginner
                #imagejets[whichJet][ibin][jbin][6] = missingouter
                #imagejets[whichJet][ibin][jbin][7] = whichVertex
        for k,v in imagejets.iteritems():
            jets.append(v)
        if len(jets) > maxToSave: break

    jetarray = np.zeros(shape=(len(jets),nBins,nBins,4))
    for i,j in enumerate(jets):
        jetarray[i] = j
    print "collected:",len(jets),"jets"
    np.save(ofname,jetarray)
    return
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
