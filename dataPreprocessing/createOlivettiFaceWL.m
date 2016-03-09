function createOlivettiFaceWL()
    load('olivettifacesoriginal.mat');
    
    DATA_LV1 = [];
    DATA_LV2 = [];
    DATA_LV3 = [];
    
    for i = 1:size(faces,2)
        
        A = reshape(faces(:,i), [64,64]);
        % make the number of columns odd
        A = A(1:end-1, 1:end-1);
        
        [cA,cH,cV,cD] = dwt2(A, 'db4');
        LV1 = [cA(:);cH(:);cV(:);cD(:)];
        
        [ccA,ccH,ccV,ccD] = dwt2(cA, 'db4');
        LV2 = [ccA(:);ccH(:);ccV(:);ccD(:)];
        size(LV2)
        
        [cccA,cccH,cccV,cccD] = dwt2(ccA, 'db4');
        LV3 = [cccA(:);cccH(:);cccV(:);cccD(:)];
        size(LV3)

        DATA_LV1 = [DATA_LV1 LV1(:)];
        DATA_LV2 = [DATA_LV2 LV2(:)];
        DATA_LV3 = [DATA_LV3 LV3(:)];
        
    end

    size(DATA_LV1)
    size(DATA_LV2)
    size(DATA_LV3)
    
    assignin('base', 'DATA_LV1', DATA_LV1);
    save('olivettifacesWL1.mat', 'DATA_LV1');
    
    assignin('base', 'DATA_LV2', DATA_LV2);
    save('olivettifacesWL2.mat', 'DATA_LV2');
    
    assignin('base', 'DATA_LV3', DATA_LV3);
    save('olivettifacesWL3.mat', 'DATA_LV3');
    
clear;
end