function testFFT()

    A = imread('yalefaces/subject01.happy');
    figure, imshow(A)
    
    F=fft2(A);   
    F2=fftshift(F);
    F2=abs(F2);
    figure,imshow(F2,[]);
    F2=log(1+F2);
    figure,imshow(F2,[]);
  
    Ahat = ifft2(F);
    figure,imshow(Ahat,[]);
    
    for i = 5:5:45
        F1 = zeros(size(F));
        h = floor(i*size(F, 1)/100):1:(100-i)*size(F, 1)/100;
        w = floor(i*size(F, 2)/100):1:(100-i)*size(F, 2)/100;
        temp = fftshift(F);
        F1(h, w) = temp(h, w); 
        F2=abs(F1);
        figure,imshow(F2,[]);

        F2norm = (F2 - min(F2(:)))/(max(F2(:)) - min(F2(:)));
        imwrite(F2norm * 255, strcat(num2str(i), 'F.jpg'), 'jpg'); 
        
        F2=log(1+F2);
        F2norm = (F2 - min(F2(:)))/(max(F2(:)) - min(F2(:)));
        figure,imshow(F2,[]);
        imwrite(F2norm, strcat(num2str(i), 'LOGF.jpg'), 'jpg'); 

        Ahat = real(ifft2(ifftshift(F1)));
        figure,imshow(Ahat,[]);
        Ahat = (Ahat - min(Ahat(:)))/(max(Ahat(:)) - min(Ahat(:)));
        imwrite(Ahat, strcat(num2str(i), 'recon.jpg'), 'jpg');
    end

end
