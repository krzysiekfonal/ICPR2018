function [Y] = preprocessing_STCM(Xraw,channel_select,dr)

% It takes a raw data, selects EMG or MMG, merges channels in the time
% domain, decimates, and then transforms to spectrograms, which are then
% filltered. 

% Wstêpna analiza wariancji
[I_1,I_2,I_3,I_4] = size(Xraw);

switch channel_select
    
    case 1
        
        channel_inx = 1:2:15; % EMG
        
    case 2
        
        channel_inx = 2:2:16; % MMG     
end

I_3 = length(channel_inx);
Xraw_new = reshape(permute(Xraw(:,:,channel_inx,:),[1 2 4 3]),[I_1 I_2 I_3*I_4]);

% Spektrogram
xd = decimate(squeeze(Xraw_new(1,1,:)),dr);
[S,Fx,Tx,Px] = spectrogram(xd,64,32,128,1e3/dr,'yaxis');
[I_4a,I_5] = size(Px);
Y = zeros([I_1 I_2 I_4a I_5]);

h = fspecial('average');
for i_1 = 1:I_1
    for i_2 = 1:I_2
             xd = decimate(squeeze(Xraw_new(i_1,i_2,:)),dr);
            [S,Fx,Tx,Px] = spectrogram(xd,64,32,128,1e3/dr,'yaxis');
             Bx = imfilter((log10(Px)), h,'replicate');
             Y(i_1,i_2,:,:) = Bx;
    end
end

% Y: class x trial x frequency x time

end