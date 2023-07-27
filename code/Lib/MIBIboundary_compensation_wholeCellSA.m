%% Yunhao Bai for MIBIsubgroup, 20200420
%compensation the boundary leakover with CODEX-like logic
%(cell B channel X counts) = (cell B channel X counts) 
%- (cell A channel X counts) * BoundaryAB/perimeterA - cell C/D
%+ (cell B channel X counts)

%assumption and approximation: 
%2. at corner of 3/4 different cells boundary, this code will triple/quadruple count that pixel.

%% 
function MIBIdataNorm = MIBIboundary_compensation_wholeCellSA(newLmod, MIBIdata, channelNormIdentity,REDSEAChecker)
%newLmod: the cell label image, boundary pixels are 0 while each cell has
%different label from 1 to cell number.
%MIBIdata: matrix of cellNum by channelNum, contains all signal counts of
%each cells.
%channelNormIdentity: passed from
%MibiExtractSingleCellDataFromSegmentation3.m, this vector labels which
%channels need to be normalized as 1, while others are 0.

[rowNum, colNum] = size(newLmod);
cellNum = max(max(newLmod));
cellPairMap = zeros(cellNum,cellNum);

%factors = -1;
niche = 1;
newLmod_padded = padarray(newLmod,[niche niche],0,'both');

%assemble the adjacency matrix
for x = 1+niche:rowNum+niche
    for y = 1+niche:colNum+niche
        if newLmod_padded(x,y) == 0
            tempM = reshape(newLmod_padded(x-1:x+1,y-1:y+1),[9 1]);
            tempFactors = unique(tempM);
            %the theoretical extreme situation is 5, in a Four Courners 
            %style, and this do appear quite frequently, so consider 3-5

            %for this part, we triple/quadruple count one pixel 3/4 times
            %for each pair.
            if length(tempFactors) >= 3 
                %only assign 1 times, as unique() sort the sequence,
                %remember that the tempFactors(2) < tempFactors(3)
                cellPairMap = cellPairMapUpdater(cellPairMap,tempFactors);
            end
        end
    end
end

%flip the matrix to make it double-direction
cellPairMap = cellPairMap+cellPairMap';

%sum the matrix to get the total boundary length of each cells
cellBoundaryTotal = sum(cellPairMap,1);

%divide by the total number of cell boundary to get the fractions
cellBoundaryTotalMatrix = repmat(cellBoundaryTotal',[1 cellNum]);
cellPairDivided = cellPairMap./cellBoundaryTotalMatrix;
cellPairDivided(isnan(cellPairDivided)) = 0; %fix replace any NaN with 0
cellPairNorm = (REDSEAChecker+1)*eye(cellNum) - cellPairDivided;

%flip the channelNormIdentity for calculation
channelNum = length(channelNormIdentity);
rev_channelNormIdentity = ones(channelNum,1) - channelNormIdentity;

%original data matrix is a cellNum by channelNum matrix, so 
MIBIdataNorm = (MIBIdata'*cellPairNorm)';
MIBIdataNorm(MIBIdataNorm<0) = 0;

%composite the normalized channels with non-normalized channels
MIBIdataNorm = MIBIdata.*repmat(rev_channelNormIdentity,[1 cellNum])' + MIBIdataNorm.*repmat(channelNormIdentity,[1 cellNum])';
end

%% auxiliary function
function cellPairMap = cellPairMapUpdater(cellPairMap,tempFactors)
%functionalize the cellPair map updating process
    cellPairs = nchoosek(tempFactors(2:end),2);
    %the function will order the pairs with elements consequentially
    for i = 1:size(cellPairs,1)
        cellPairMap(cellPairs(i,1),cellPairs(i,2)) = cellPairMap(cellPairs(i,1),cellPairs(i,2)) + 1;
    end
end