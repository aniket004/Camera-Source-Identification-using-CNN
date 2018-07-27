
% take cropped images

% take path of image folder
dr = uigetdir();
list = dir([dr,'/*.JPG']);
num_img = length(list);


total_img = zeros(num_img*256,256);

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
% for x = 1:num_img
%     x
%     imagepath = [dr,'/',list(x).name];
%     img = imread(imagepath);
%     
%     crop_img = imcrop(img,[1 1 255 255]);
%     total_img((256*(x-1)+1):256*x,1:256) = crop_img;       
% end

filename_label = 'Dresden_10_cam_100_gray_img_label.csv';
csvwrite(filename_label,label);

% filename_img = 'Dresden_10_cam_100_gray_img_126_crop_img.csv';
% csvwrite(filename_img,total_img);
