netname = "alexnet";
net = alexnet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);
%%
layerNames = {dlnet.Layers.Name}; % ["conv2d_1", "conv2d_2", "conv2d_3"]; % Specify the layer names
% inputSize = [256, 256, 3]; % Specify the input size
inputSize = net.Layers(1).InputSize;
repN = 100; % Specify the number of input images
device = 'gpu'; % Specify the device ('cpu' or 'gpu')
tic
[RFmaps, fig] = mapReceptiveField(dlnet, layerNames, inputSize, repN, device);
toc
exportgraphics(gcf,netname+"_rf.png")
exportgraphics(gcf,netname+"_rf.pdf")
save(netname+"_RFmaps.mat","RFmaps")