function [layerShapes, layerShapesMap] = get_layer_shape(dlnet, layerNames, dlImgs)
%get_layer_shape Returns the sizes of specific layers in a deep learning network.
%   
%   [layerShapes, layerShapesMap] = get_layer_shape(dlnet, layerNames, dlImgs)
%   takes a dlnetwork object (dlnet), a formatted images input (dlImgs),
%   and a cell array of layer names (layerNames). It returns a table (layerShapes)
%   and a dictionary (layerShapesMap) mapping each layer name to its output size.
%
%   Inputs:
%   - dlnet: dlnetwork object
%   - layerNames: Cell array of strings, each string is a layer name from dlnet
%                 This input is optional, if it's not provided, then it will compute 
%                 shape of all layers in the dlnet.
%   - dlImgs: Formatted images for dlnetwork prediction, usually rand of 255 scale.
%
%   Outputs:
%   - layerShapes: A table with each row corresponding to a layer. The first column
%     contains the layer name and the second column contains the size of the
%     layer's output.
%   - layerShapesMap: A containers.Map object (dictionary). Each key is a layer name
%     and its value is the size of the layer's output.
%
%   Example:
%   [layerShapes, layerShapesMap] = get_layer_shape(dlnet, {'layer1','layer2'}, dlImgs);
%   layerShapes = get_layer_shape(dlnet,{dlnet.Layers.Name},dlImg);
%   layerShapes = get_layer_shape(dlnet,{dlnet.Layers.Name});
%   layerShapes = get_layer_shape(dlnet);
    if nargin <= 1
        layerNames = {dlnet.Layers.Name};
    end
    if nargin <= 2
        img = rand(256,256,3,1)*255.0;
        dlImgs = dlarray(gpuArray(single(img)),'SSCB');
        fprintf("Using images of size [256,256,3,1] as input.\n")
    end
    scores_out = cell(1,numel(layerNames));
    [scores_out{:}] = predict(dlnet,dlImgs,'Outputs',layerNames);
    layerNames = reshape(layerNames,[],1);
    layerSizes = cell(numel(layerNames),1);
    layerShapesMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:numel(layerNames)
        layerSizes{i} = size(scores_out{i});
        layerShapesMap(layerNames{i}) = size(scores_out{i});
    end 
    layerShapes = table(layerNames, layerSizes, 'VariableNames', {'LayerName', 'Size'});
end