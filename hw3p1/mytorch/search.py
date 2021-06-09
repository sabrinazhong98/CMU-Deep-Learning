

def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)


    symbol,seqlen,batch_size = y_probs.shape
    forward_path = []
    forward_prob = []
    
    
    for b in range(batch_size):
        probs = 1
        paths = ['empty']*seqlen

        for s in range(seqlen):
            cmax = 0
            current = 'empty'
            
            for m in range(symbol):
                if y_probs[m][s][b] > cmax:
                    cmax = max(y_probs[m][s][b],cmax)

                    if m == 0:
     
                        current = 'empty'
                    else:
                        current = SymbolSets[m-1]
                   
            paths[s] = current
            probs = probs * cmax
        forward_path.append(paths)
        forward_prob.append(probs)
 
    
    forward_path_compress = []
    for b in range(batch_size):
       compressed = ""
       prev = False
       for s in range(seqlen):
           single = forward_path[b][s]
           
           if single == 'empty':
               prev = False
     
           elif single != prev:
               compressed += single
               prev = single
                              
       forward_path_compress.append(compressed)
    return forward_path_compress[0], forward_prob[0]


##############################################################################



    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)



blank = 0

def InitializePaths(SymbolSet, y):
    InitialBlankPathScore = {}
    InitialPathScore = {}
    path = ""
    InitialBlankPathScore[path] = y[int(blank)]
    InitialPathsWithFinalBlank = {path}
    
    InitialPathsWithFinalSymbol = set()
    for c in range(len(SymbolSet)):
        path = SymbolSet[c]
        InitialPathScore[path] = y[c+1]
        InitialPathsWithFinalSymbol.add(path)
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathWithTerminalBlank,PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathWithTerminalBlank = set()
    UpdatedBlankPathScore = {}
    for path in PathWithTerminalBlank:
        UpdatedPathWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[blank]
    for path in PathsWithTerminalSymbol:
        if  path in UpdatedPathWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[blank]
        else:
            UpdatedPathWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[blank]
    return UpdatedPathWithTerminalBlank,UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet,y, BlankPathScore, PathScore ):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}
    for path in PathsWithTerminalBlank:
        for c in range(len(SymbolSet)):
            newpath = path + SymbolSet[c]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[c+1]
    
    for path in PathsWithTerminalSymbol:
        for c in range(len(SymbolSet)):
            if SymbolSet[c] == path[-1]:
                
                newpath = path 
            else:
                newpath = path + SymbolSet[c]
            
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] += PathScore[path] *y[c+1]
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] *y[c+1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
   
    scorelist = []
    # First gather all the relevant scores
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist.sort(reverse=True)
    if BeamWidth < len(scorelist):
        cutoff = scorelist[BeamWidth]
    else:
        cutoff = scorelist[-1]
    
    PrunedPathsWithTerminalBlank= set()
    
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]
            
    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):

    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    

    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore


def BeamSearch(SymbolSet, y, BeamWidth):
   
    PathScore = {}
    BlankPathScore = {}
    symbol,seqlen,batch_size = y.shape
    
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore,NewPathScore = InitializePaths(SymbolSet, y[:,0,:])
    

    for t in range(1, seqlen):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                           NewPathsWithTerminalSymbol,
                                                                                           NewBlankPathScore, NewPathScore,
                                                                                           BeamWidth)

        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y[:,t,:], BlankPathScore, PathScore)

        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y[:, t, :], BlankPathScore, PathScore)

    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)

    allpath = sorted(FinalPathScore.items(), key = lambda x:x[1], reverse = True)
    BestPath = allpath[0][0]
    
    return BestPath, FinalPathScore
