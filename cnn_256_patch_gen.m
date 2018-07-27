% take cropped images

% take path of image folder
dr = uigetdir();
list = dir([dr,'/*.JPG']);
num_img = length(list);


%total_img = zeros(num_img*256*70,256);

total_img_block = []; 

for x = 1:num_img
    if( strmatch('Canon', list(x).name) == 1 )
        label(x) = 0;
    elseif( strmatch('Olympus', list(x).name) == 1 )
        label(x) = 1;
    elseif( strmatch('Samsung', list(x).name) == 1 )
        label(x) = 2;
    elseif( strmatch('Sony', list(x).name) == 1 )
        label(x) = 3;
    elseif( strmatch('Agfa', list(x).name) == 1 )
        label(x) = 4;
    elseif( strmatch('Casio', list(x).name) == 1 )
        label(x) = 5;
    elseif( strmatch('FujiFilm', list(x).name) == 1 )
        label(x) = 6;
    elseif( strmatch('Kodak', list(x).name) == 1 )
        label(x) = 7;
    elseif( strmatch('Nikon', list(x).name) == 1 )
        label(x) = 8;
    elseif( strmatch('Panasonic', list(x).name) == 1 )
        label(x) = 9;
    end
end
for x = 1:num_img
    x
    imagepath = [dr,'/',list(x).name];
    img = imread(imagepath);
    
    crop_img = imcrop(img,[1 1 (256*7-1) (256*10-1)]);  %%image size 256*10X256*7
    
    for i = 1:7
        for j = 1:7
            i*j
            block_img = crop_img((256*(i-1)+1):256*i,(256*(j-1)+1):256*j);
            total_img_block = vertcat(total_img_block,block_img); 
            %total_img((70*256*(x-1)+256*(i-1)+256*(j-1)+1):70*256*x+256*i+256*j,1:256) = block_img;    
        end
    end
    
    %total_img((256*(x-1)+1)*i*j:256*x*i*j,1:256) = block_img(i,j);    
    total_label(49*(x-1)+1:49*x,1) = label(x);
end

filename_label = 'Dresden_10_cam_100_gray_256_patch_img_label.csv';
csvwrite(filename_label,total_label);

filename_img = 'Dresden_10_cam_100_gray_img_256_patch_crop_img.csv';
csvwrite(filename_img,total_img_block);
