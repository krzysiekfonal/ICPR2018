function [Y] = preprocessing_Spect_5D(Xraw,channel_select)

% It takes a raw data, selects EMG or MMG, and transforms to spectrograms, 
%which are then filltered. 

% Wstêpna analiza wariancji
[I_1,I_2,I_3,I_4] = size(Xraw);

switch channel_select
    
    case 1
        
        channel_inx = 1:2:15; % EMG
        
    case 2
        
        channel_inx = 2:2:16; % MMG     
end

% Wstêpny spektrogram
[S,Fx,Tx,Px] = spectrogram(squeeze(Xraw(1,2,3,:)),64,32,128,1e3,'yaxis');
[I_4a,I_5] = size(Px);
Y = zeros([I_1 I_2 length(channel_inx) I_4a I_5]);

% Filtracja uœredniaj¹ca 
h = fspecial('average');

% Tworzenie tensora spektrogramów
for i_1 = 1:I_1
    for i_2 = 1:I_2
        for i = 1:length(channel_inx) 
            i_3 = channel_inx(i);
            [S,Fx,Tx,Px] = spectrogram(squeeze(Xraw(i_1,i_2,i_3,:)),64,32,128,1e3,'yaxis');
            Bx = imfilter(log10(Px), h,'replicate');
          %  Bx = imfilter((log10(Px) + 6), h,'replicate');
%            if min(min(Bx)) < 0
%               Bx = Bx - min(min(Bx)) + eps; 
%            end
            Y(i_1,i_2,i,:,:) = Bx;
        end
    end
end

% Y: class x trial x channel x frequency x time

end