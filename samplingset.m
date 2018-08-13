function [Xtrain , ytrain , Xtest , ytest , Xvalid , yvalid] = samplingset(X , y , Itrain , Itest , i , Ivalid);


if (nargin < 5)
    
    i    = ceil(size(Itrain , 1)*rand);
    
end

if (nargin < 6)
    
    Xvalid = [];
    
    yvalid = [];
    
end

if ((i < 1) || (i > size(Itrain , 1)))
    
    error('index i is not valid');
    
end

indtrain    = Itrain(i , :);

Xtrain      = X(: , indtrain);

ytrain      = y(indtrain);

indtest     = Itest(i , :);

Xtest       = X(: , indtest);

ytest       = y(indtest);

if (nargin == 6)

    indvalid     = Ivalid(i , :);

    Xvalid       = X(: , indvalid);

    yvalid       = y(indvalid);

end
