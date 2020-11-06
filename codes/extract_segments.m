%%Extract cardiac segments from heart sound signals

clc
clear all
close all

%% Initialize Parameters

max_audio_length = 20;    %seconds
N = 60;                   %order of filters
sr = 1000;                %resampling rate
nsamp = 2500;           %number of samples in each cardiac cycle segment
X = [];
Y = [];
file_name = [];
states = [];

%% Initialize paths

datapath= '../data/kaggle/wav_files/';
savedir='../data/kaggle/feature/';
labelpath = '../data/kaggle/label/';
avoid_filter = false; % false - with FIR, true - withour FIR

addpath(genpath('matlabUtils/'));

%% Import annotations

anno = importdata(labelpath + "Annotation.csv");
labels = anno.data(:, 3:11);
num_files = anno.data(:, 2);

%% Initialize filter bank

if(~avoid_filter)
    Wn = 45*2/sr; % lowpass cutoff
    b1 = fir1(N,Wn,'low',hamming(N+1));
    Wn = [45*2/sr, 80*2/sr]; %bandpass cutoff
    b2 = fir1(N,Wn,hamming(N+1));
    Wn = [80*2/sr, 200*2/sr]; %bandpass cutoff
    b3 = fir1(N,Wn,hamming(N+1));
    Wn = 200*2/sr; %highpass cutoff
    b4 = fir1(N,Wn,'high',hamming(N+1));
end

%% Feature extraction using Springers Segmentation

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;

%% Extract Features

for sub = 1 : length(num_files) % avoid using the first ones
    for pos = 1 : num_files(sub)
        fname = strcat(num2str(sub), "_", num2str(pos), ".wav");
        [PCG, Fs1] = audioread(strcat(datapath, num2str(sub), "/", fname));
        if length(PCG) > max_audio_length*Fs1
            PCG = PCG(1 : max_audio_length*Fs1); % Clip signals to max_audio_length seconds
        end
        
        %--- Pre-processing (resample + bandpass + spike removal)
        
        % resample to 1000 Hz
        PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); 
        % filter the signal between 25 to 400 Hz
        PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
        PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
        % remove spikes
        PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);

        %--- Run springer's segmentation

        assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                        springer_options.audio_Fs,... 
                        Springer_B_matrix, Springer_pi_vector,...
                        Springer_total_obs_distribution, false);

        [idx_states , last_idx]=get_states(assigned_states); %idx_states ncc x 4 matrix 
                                    % containing starting index of segments 
        
        ncc = size(idx_states,1);
        if(avoid_filter)
            PCG = PCG_resampled;
            x = nan(ncc,nsamp);
        else
            % Dividing signals into filter banks
            clear PCG
            PCG(:,1) = filtfilt(b1,1,PCG_resampled);
            PCG(:,2) = filtfilt(b2,1,PCG_resampled);
            PCG(:,3) = filtfilt(b3,1,PCG_resampled);
            PCG(:,4) = filtfilt(b4,1,PCG_resampled);
            nfb = 4;
            x = nan(ncc,nsamp,nfb);
        end

        if(avoid_filter)
            for row=1:ncc
                if row == ncc % for the last complete cardiac cycle
                    tmp = PCG(idx_states(row,1):last_idx-1);
                else
                    tmp = PCG(idx_states(row,1):idx_states(row+1,1)-1);
                end
                N = nsamp-length(tmp); % append zeros at the end of cardiac cycle
                x(row,:) = [tmp; zeros(N,1)];
                file_name=[file_name;string(fname)]; % matrix containing the filename
                                                        % of each cardiac cycle
                Y=[Y;labels(sub,:)];               % Class labels for each cardiac cycle
            end
        else
            for row=1:ncc
                for fb=1:nfb
                    if row == ncc % for the last complete cardiac cycle
                        tmp = PCG(idx_states(row,1):last_idx-1,fb);
                    else
                        tmp = PCG(idx_states(row,1):idx_states(row+1,1)-1,fb);
                    end
                    N = nsamp-length(tmp); % append zeros at the end of cardiac cycle
                    x(row,:,fb) = [tmp; zeros(N,1)];
                end
                file_name=[file_name;string(fname)]; % matrix containing the filename
                                                    % of each cardiac cycle
                Y=[Y;labels(sub,:)];               % Class labels for each cardiac cycle
            end
        end
        X=[X;x]; % matrix containing all cardiac cycles
        states=[states;idx_states]; % matrix containing 
                                    %index of states of each cardiac cycle
        
    end
end

%% Save Data

if(avoid_filter)
    file = 'feature_noFIR.mat';
else
    file = 'feature.mat';
end
sname=strcat(savedir, file); 
save(sname, 'X', 'Y', 'states', 'file_name');

%% function to extract state index

function [idx_states,last_idx] = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end); % K controls the starting cycle
                                        % of the segment. Starting cycle
                                        % is always kept 1 through the 
                                        % switch cases (!)
                                        
    rem                  = mod(length(indx2),4);
    last_idx             = length(indx2)-rem+1;
    indx2(last_idx:end) = []; % clipping the partial segments in the end
    idx_states           = reshape(indx2,4,length(indx2)/4)'; % idx_states 
                            % reshaped into a no.segments X 4 sized matrix
                            % containing state indices
end
