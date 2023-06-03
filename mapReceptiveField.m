function [RFmaps, fig] = mapReceptiveField(dlnet, layerNames, inputSize, repN, device)
% mapReceptiveField calculates and visualizes the receptive field maps of specified layers in a network.
%
% Inputs:
%   - dlnet: The deep learning network (dlnetwork object) to analyze.
%   - layerNames: A cell array of layer names for which to compute the receptive field maps.
%   - inputSize: The size of the input images in the format [height, width, channels].
%   - repN: The number of random input images to use for calculating the receptive field maps.
%   - device: The device to perform the computations on ('cpu' or 'gpu').
%
% Outputs:
%   - RFmaps: A structure containing the receptive field maps. Each map is stored under its corresponding layer name.
%   - fig: The figure handle of the visualization showing the receptive field maps.
%
% Example usage:
%   netname = "inceptionv3";
%   net = inceptionv3;
%   lgraph = layerGraph(net);
%   lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
%   dlnet = dlnetwork(lgraph);
% 
%   layerNames = ["conv2d_1", "conv2d_2", "conv2d_3"]; % Specify the layer names
%   inputSize = [256, 256, 3]; % Specify the input size
%   repN = 100; % Specify the number of input images
%   device = 'gpu'; % Specify the device ('cpu' or 'gpu')
% 
%   [RFmaps, fig] = mapReceptiveField(dlnet, layerNames, inputSize, repN, device);
%
% Note:
%   The function requires the 'gradientMap_batch' function to be defined and accessible in the workspace.
%   It is assumed that the necessary supporting functions and dependencies are available.
%
    % Check if inputSize is provided, otherwise assign default value
    if nargin < 3, inputSize = [256, 256, 3]; end
    % Check if repN is provided, otherwise assign default value
    if nargin < 4, repN = 100; end
    % Check if device is provided, otherwise assign default value
    if nargin < 5, device = 'gpu'; end
    % Create empty structure to store receptive field maps
    RFmaps = struct();
    
    % Set device to CPU or GPU
    if strcmpi(device, 'gpu')
        dlarrayFn = @(x) dlarray(gpuArray(single(x)), 'SSCB');
    else
        dlarrayFn = @(x) dlarray(single(x), 'SSCB');
    end
    
    [layerShapesTab,layerShapeMap] = get_layer_shape(dlnet, layerNames, dlarrayFn(rand(inputSize)*255.0));
    % Create tiled layout for visualization
    fig = figure;
    set(gcf, 'WindowState', 'maximized');
    T = tiledlayout("flow", 'pad', 'tight', 'TileSp', 'tight');
    
    % Iterate over layer names
    for layerIdx = 1:numel(layerNames)
        layerName = layerNames{layerIdx};
        disp(layerName)

        % Generate random input images
        img = rand(inputSize(1), inputSize(2), inputSize(3), repN) * 255.0;
        dlImg = dlarrayFn(img);

        % Get layer shape
        % layerShape = size(predict(dlnet, dlImg, 'Outputs', layerName));
        layerShape = layerShapeMap(layerName);

        % Calculate gradient map based on layer shape
        if numel(layerShape) == 4
            % If it's a 4D layer (conv), calculate the gradient map for all channels
            % at the center pixel
            cent_i = max(1, floor(layerShape(1) / 2));
            cent_j = max(1, floor(layerShape(2) / 2));
            chan_id = 1:layerShape(3);
            dydI = dlfeval(@gradientMap_batch, dlnet, dlImg, layerName, chan_id, cent_i, cent_j);
        elseif numel(layerShape) == 2
            % If it's a 2D layer, calculate the gradient map for all channels
            chan_id = 1:layerShape(1);
            dydI = dlfeval(@gradientMap_batch, dlnet, dlImg, layerName, chan_id);
        end
        
        % Extract and store the gradient map
        gradmap = extractdata(mean(abs(dydI), [3, 4]));
        RFmaps.(layerName) = gradmap;
        
        % Plot the gradient map
        ax = nexttile(T);
        imagesc(gradmap);
        title(layerName, 'Interpreter', 'none');
        axis image;
        axis off;
    end
    
    % Configure tiled layout
    % T = findobj(fig, 'Type', 'TiledLayout');
end