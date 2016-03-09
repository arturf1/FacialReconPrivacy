function decodeYaleFaceFFT(V)

R = V(1:319*122);
theta = V(319*122+1:end);
R = reshape(R, [122, 319]);
theta = reshape(theta, [122, 319]);
FFTimg = R.*exp(1i*theta);
FFTimg = [FFTimg; conj(rot90(FFTimg(1:121,:),-2))];       
img = ifft2(ifftshift(FFTimg));
figure,imshow(img,[]);

end
