function decodeOlivettiFaceFFT(V)

R = V(1:63*32);
theta = V(63*32+1:end);
R = reshape(R, [32, 63]);
theta = reshape(theta, [32, 63]);
FFTimg = R.*exp(1i*theta);
FFTimg = [FFTimg; conj(rot90(FFTimg(1:31,:),-2))];       
img = ifft2(ifftshift(FFTimg));
figure,imshow(img,[]);

end
