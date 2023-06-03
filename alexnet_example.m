%% Load alexnet and modify it
net = alexnet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);
%% Get layer shape for all layers
layerShapes = get_layer_shape(dlnet);
disp(layerShapes)
%% Get layer shape for all layers
img = rand(256,256,3,1)*255.0;
dlImgs = dlarray(single(img),'SSCB');
[layerShapes,layerShapeMap] = get_layer_shape(dlnet,{dlnet.Layers.Name},dlImgs);
disp(layerShapes)
%%
img = rand(256,256,3,100)*255.0;
dlImg = dlarray(single(img),'SSCB');
dydI = dlfeval(@gradientMap_batch,dlnet,dlImg,'fc6',355);
gradmap = extractdata(mean(abs(dydI),[3,4]));
figure;imagesc(gradmap);axis image
%%
img = rand(256,256,3,100)*255.0;
dlImg = dlarray(single(img),'SSCB');
dydI = dlfeval(@gradientMap_batch,dlnet,dlImg,'conv2',1:256,15,15);
gradmap = extractdata(mean(abs(dydI),[3,4]));
figure;imagesc(gradmap);axis image
%%
% another way of computing the map grad * image
gradimgmap = extractdata(mean(abs(dydI .* dlImg),[3,4]));
figure;imagesc(gradimgmap);axis image
%% all layers in alexnet
figure;
tiledlayout("flow",'pad','tight','TileSp','tight');
repN = 100;
for rowi = 1:size(layerShapes,1)
layerShape = layerShapes.Size{rowi};
layerName = layerShapes.LayerName{rowi};
disp(layerName)
img = rand(256,256,3,repN)*255.0;
dlImg = dlarray(single(img),'SSCB');
if length(layerShape)==4
    % If it's a 4D layer (conv), calculate the gradient map for all channels
    % at the center pixel
    cent_i = floor(layerShape(1)/2);
    cent_j = floor(layerShape(2)/2);
    chan_id = 1:layerShape(3);
    dydI = dlfeval(@gradientMap_batch,dlnet,dlImg,layerName,chan_id,cent_i,cent_j);
elseif length(layerShape)==2
    % If it's a 2D layer, calculate the gradient map for all channel
    chan_id = 1:layerShape(1);
    dydI = dlfeval(@gradientMap_batch,dlnet,dlImg,layerName,chan_id);
end
gradmap = extractdata(mean(abs(dydI),[3,4]));
ax = nexttile;imagesc(gradmap);title(layerName);axis image;
end