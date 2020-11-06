clear
clc
rng('default')
fold = 1; % number of normal/abnormals in new folds
datapath='../data/kaggle/feature/';
fold_text_path='../data/kaggle/feature/folds/text/';
fold_save='../data/kaggle/feature/folds/';
avoid_filter = true; % false - with FIR, true - withour FIR

%% Load annotations and mat files

anno_file = '../data/kaggle/label/Annotation.csv';
anno = importdata(anno_file);
num_files = anno.data(:, 2);

if(avoid_filter)
    load([datapath 'feature_noFIR.mat']);
else
    load([datapath 'feature.mat']);
end

%% Start creating folds

trainX=[];
trainY=[];
valX=[];
valY=[];
filestrain=[];
filesval=[];
cc_train=[];
cc_val=[];
train_parts=[];
val_parts=[];
    
%% Partition training data

cwd = pwd;
loadfile=[cwd(1:end-6) fold_text_path(3:end) 'train.txt'];% creating full path to avoid file exception
train=sort(importdata(loadfile));
for idx=1:length(train)
    for pos = 1:num_files(train(idx))
        name = strcat(num2str(train(idx)), "_", num2str(pos), ".wav");
        trainX = [trainX;X(file_name==name,:,:)];
        trainY = [trainY;Y(file_name==name,:)];
        filestrain = [filestrain; file_name(file_name==name)];
        cc_train  = [cc_train;states(file_name==name)];
        train_parts = [train_parts;sum(file_name==name)];
    end
end
filestrain = convertStringsToChars(filestrain);

%% Partition validation data

cwd = pwd;
loadfile=[cwd(1:end-6) fold_text_path(3:end) 'validation.txt'];
val=sort(importdata(loadfile));
for idx=1:length(val)
    for pos = 1:num_files(val(idx))
        name = strcat(num2str(val(idx)), "_", num2str(pos), ".wav");
        valX = [valX;X(file_name==name,:,:)];
        valY = [valY;Y(file_name==name,:)]; 
        filesval = [filesval; file_name(file_name==name)];
        cc_val  = [cc_val;states(file_name==name)];
        val_parts = [val_parts;sum(file_name==name)];
    end
end  
filesval = convertStringsToChars(filesval);

%% Save data

if(avoid_filter)
    save_name = ['fold_k_noFIR.mat'];
else
    save_name = ['fold_k.mat'];
end

disp(['saving' ' ' save_name])
%     clearvars -except cc_train train_parts cc_val val_parts
save([fold_save save_name], 'trainX', 'trainY', 'valX', 'valY', 'cc_train',...
        'train_parts', 'cc_val', 'val_parts', 'filestrain', 'filesval', '-v7.3');
