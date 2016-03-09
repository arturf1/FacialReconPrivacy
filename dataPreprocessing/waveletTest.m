function waveletTest()

    A = imread('yalefaces/subject01.happy');
    figure, imshow(A)
    
    [cA,cH,cV,cD] = dwt2(A, 'db4');
    [ccA,ccH,ccV,ccD] = dwt2(cA, 'db4');
    [cccA,cccH,cccV,cccD] = dwt2(ccA, 'db4');
    figure,imshow(cccA,[]);
    figure,imshow(cccH,[]);
    figure,imshow(cccV,[]);
    figure,imshow(cccD,[]);
    
    X = idwt2(cA,cH,cV,cD,'db4')
    figure,imshow(X,[]);
end