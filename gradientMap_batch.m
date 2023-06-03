function dydI = gradientMap_batch(dlnet, dlImgs, layerName, classIdx, pos_i, pos_j)
%gradientMap_batch Compute the gradient of a class score with respect to input images in batch
%
%   dydI = gradientMap_batch(dlnet, dlImgs, layerName, classIdx, pos_i, pos_j)
%   takes a dlnetwork object (dlnet), a formatted images input (dlImgs),
%   a layer name (layerName), a class index (classIdx), and coordinates 
%   (pos_i, pos_j). It computes the gradient of the class score with 
%   respect to the input images and returns a dlarray (dydI) containing
%   these gradients.
%
%   Inputs:
%   - dlnet: dlnetwork object.
%   - dlImgs: Formatted images for dlnetwork prediction.
%   - layerName: String, the name of the layer from dlnet.
%   - classIdx: Index of the class for which the gradient is calculated.
%   - pos_i: Index for the height of the image (required for conv layers).
%   - pos_j: Index for the width of the image (required for conv layers).
%
%   Outputs:
%   - dydI: dlarray, the gradients of the class score with respect to the 
%     input images.
%
%   Example:
%   net = alexnet;
%   lgraph = layerGraph(net);
%   lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
%   dlnet = dlnetwork(lgraph);
%     img = rand(256,256,3,100)*255.0;
%     dlImg = dlarray(single(img),'SSCB');
%     dydI = dlfeval(@gradientMap_batch,dlnet,dlImg,'conv2',1:256,15,15);
%     gradmap = extractdata(mean(abs(dydI),[3,4]));
%     figure;imagesc(gradmap);axis image
scores = predict(dlnet,dlImgs,'Outputs',{layerName});
NDIM = ndims(scores);
if NDIM == 2 || NDIM == 1 % fc
    activation = scores(classIdx, :);
elseif NDIM == 3 || NDIM == 4 % conv
    disp(size(scores))
    activation = scores(pos_i, pos_j, classIdx, :);
    disp(size(activation))
else
    disp(size(scores))
    error("Shape of scores not expected")
end
activation_sum = sum(activation,'all'); % sum over batch and all remaining channels
dydI = dlgradient(activation_sum,dlImgs);
end