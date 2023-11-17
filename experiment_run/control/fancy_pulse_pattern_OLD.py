
# This is not currently in use at ANU, but may be useful one day.
def generateComplexPulseMagPattern(pulseSeqDesc, pulseTimeParams, 
        heightParams,
        sampleRate=2500000
    ):
    """Generate drive waveforms for magnetic pulsing along 3 axes.

    pulseSeqDesc: a list of tuples describing the presence (or not) of pulses.
    E.g. [("X","Y"), ("mX","mY")] is a a +X pulse first followed by a Y pulse in the first sequence, then a -X pulse followed by a -Y pulse in the second. The timing of the pulses is according to parameters in pulseTimeParams, and the size of the pulses comes from heightParams. sampleRate is the number of samples per second of the FG.
    An extra 'reversing' pulse is also included at the end of each sequence, to reset
    the atoms to where they started for the next pump pulse.
    Another option is 'depol' which can use a strong field in a given direction to remove unwanted offsets due to imperfect pulses or alignment. It needs some polishing though.
    """
    Nseq=len(pulseSeqDesc)
    hp=heightParams
    bonusZHeight=hp.bonusZHeight
    d1=dict( #First pulse
        X=(hp.piX, hp.piXOff,0),
        mX=(hp.mPiX, hp.mPiXOff,0),
        Y=(hp.piYOff, hp.piY, 0),
        mY=(hp.mPiYOff, hp.mPiY, 0.),
        X_B=(hp.piX_B, hp.piXOff,0),
        mX_B=(hp.mPiX_B, hp.mPiXOff,0),
        Y_B=(hp.piYOff, hp.piY_B, 0),
        mY_B=(hp.mPiYOff, hp.mPiY_B, 0.),
        
    )
    d1[''] = (0,0,0) #"No pulse"
    d1['0'] = (0,0,0)

    if 1:
        d2=dict( #Second pulse
            X=( hp.pi2X if hp.pi2X is not None else (hp.piX-hp.mPiX)/2., 0,hp.pi2XOff),
            mX=( -hp.pi2X if hp.pi2X is not None else -(hp.piX-hp.mPiX)/2.,0, -hp.pi2XOff),
            Y=( 0, hp.pi2Y if hp.pi2Y is not None else (hp.piY-hp.mPiY)/2., hp.pi2YOff),
            mY=(0, -hp.pi2Y if hp.pi2Y is not None else -(hp.piY-hp.mPiY)/2., -hp.pi2YOff),
    #y=log(exp(t*1e2) ) -log(log( exp(t*1e2) ))
        )
        d2['']=(0.,0.,0.)
        d2['0']=(0.,0.,0.)
        d2copy=d2.copy()
        for k in d2copy.keys():
            d2['s'+k]=d2copy[k]
    print("Bonus z height: {}".format(bonusZHeight))
    d3=dict( #Bonuz Z-pulse
        X=tuple(bonusZHeight),
        mX=tuple(bonusZHeight),
        Y=tuple(bonusZHeight),
        mY=tuple(bonusZHeight),
        X_B=tuple(bonusZHeight),
        mX_B=tuple(bonusZHeight),
        Y_B=tuple(bonusZHeight),
        mY_B=tuple(bonusZHeight),
        )
    d3['']=(0,0,0)
    d3[0]=(0,0,0)


    t0,tLong, tauLong, width,longWidth, pi2RelWidth, tShort, tauShort, depolLength, tInitDepol=pulseTimeParams.t0,pulseTimeParams.tLong,pulseTimeParams.tau, pulseTimeParams.width, pulseTimeParams.longWidth, pulseTimeParams.pi2RelWidth, pulseTimeParams.tShort, pulseTimeParams.tauShort, pulseTimeParams.depolLength, pulseTimeParams.tInitDepol
    
    startTimes=[]
    pulseWidths=[]

    initDepol = 0.0 if tInitDepol==0 else 1.0
    pulseHeights=[  (   d1[plsPair[0]], 
                        [initDepol*v for v in d2[plsPair[1]]],
                        d3[plsPair[0]], #bonus z-pulse
                        [1*v for v in d2[plsPair[1]]],
                        [heightParams.reverseHeight*v for v in d1[plsPair[0]]] #undoing
                    ) for plsPair in pulseSeqDesc]

    #Replace with something similar to the pulseHeights one
    tOffs=0
    for k in range(Nseq):
        #Short trace-------------
        bShortTrace=pulseSeqDesc[k][-1].startswith('s')
        thisTau= tauShort if bShortTrace else tauLong
        thisTTotal=tShort if bShortTrace else tLong

        #tOffs=(tLong+tShort)*k
        if tShort>0 and 0:
            startTimes.extend([t0+tOffs, t0+tauShort+tOffs])
            pulseWidths.extend([width, width])
            tOffs+=tShort

        #Long trace-------------
        #Could add in bonus Bz pulse here
        startTimes.extend([t0+tOffs, 
                        t0+tOffs+tInitDepol,
                        t0+tOffs+thisTau-width*2, #Bz pulse
                        t0+tOffs+thisTau,
                        tOffs+thisTTotal-width-40e-6, #revert pulse
                        ])
        pulseWidths.extend([width, 
                longWidth, #initDepol
                1*width, #Bonus Z
                pi2RelWidth*(width if tInitDepol==0 else longWidth),
                width
                ])
        tOffs += thisTTotal


    pulseHeights=np.array(pulseHeights).reshape(-1,3)
  

    tLast=np.max(np.array(startTimes)+np.array(pulseWidths)) 
    print("latest thing in sequence: {}".format(tLast) )
    tSeqTotal=tLast+10e-6
    sampleRate=np.array(sampleRate)
    if sampleRate.size==1:
        sampleRate=np.ones(3, dtype='f8')*sampleRate
    #Nsamples=32768*2
    tSeqTotal=tLast+10e-6
    Nsamples=(tSeqTotal*sampleRate).astype('i4')
    #tSeqTotal=int(tSeqTotal*sampleRate)/sampleRate
    #Nsamples=int(tSeqTotal*sampleRate)
    print("Nsamples: {}".format(Nsamples) )
    #sampleRate=Nsamples/tSeqTotal

    tx,y1=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,0],
                            sampleRate=sampleRate[0],
                            Nsamples=Nsamples[0],
                            )
    ty,y2=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,1],
                            sampleRate=sampleRate[1],
                            Nsamples=Nsamples[1],
                            )
    tz,y3=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,2],
                            sampleRate=sampleRate[2],
                            Nsamples=Nsamples[2],
                            )
    return tx, y1, y2,y3
