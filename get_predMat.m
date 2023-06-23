function [P,S] = get_predMat(A, B, Np)
    P = [A^0;A^1;A^2;A^3;A^4;A^5];
    zz = zeros(size(B));
    S = [zz,zz,zz,zz,zz;
        B,zz,zz,zz,zz;
        A*B, B,zz,zz,zz;
        A^2*B,A*B, B,zz,zz;
        A^3*B,A^2*B,A*B, B,zz;
        A^4*B,A^3*B,A^2*B,A*B, B;];
end