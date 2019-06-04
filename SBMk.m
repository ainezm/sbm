function [V] = SBMk(n,a,b)

k=2;

A = triu(rand(k*n,k*n));
A(A>b/n) = 0;
A = sparse(A);

for i=1:k
    A11 = triu(rand(n));
    A11(A11>a/n) = 0; A11 = sparse(A11);
    id = ((i-1)*n+1):i*n; id = id';
    A(id,id) = A11;
end

A = triu(A,1); 
A = sparse(A+A'); A(A>0) = 1;

tic;

[ai,aj,av] = find(triu(A));
m= length(ai);

% creating a random splitting of the edges
edges = rand(m,1);
edges1 = find(edges<=0.5);
edges2 = find(edges>0.5);
A = sparse(ai(edges1),aj(edges1),ones(size(edges1)),k*n,k*n);
B = sparse(ai(edges2),aj(edges2),ones(size(edges2)),k*n,k*n);
A = A + A';
B = B + B';
aa = A; bb = B;


r = rand(k*n,1);
r = r>0.5;
Y = find(r);
Z = setdiff(1:k*n,Y);

r = rand(length(Y),1);
r = r>0.5;
Y1 = Y(find(r));
Y2 = setdiff(Y,Y1);

A1 = A(Z,Y1);

[U,S,V] = svds(A1,k);

l=k; %3*k;
colsY2 = Y2([ceil(n/8),ceil(3*n/8)]);  %(Y2(randperm(length(Y2),l));
A2 = A(Z,colsY2) - (a+b)/(4*n);   

projY2 = U*U'*A2;

VV = zeros(l,n/2);

for i=1:l
   e1 = projY2(:,i); 
   [vec, idx]  =sort(e1,'descend');
   VV(i,:) = Z(idx(1:n/2));
end


%outC1 = setdiff(VV(1,:),intersect(VV(1,:),union(VV(2,:),VV(3,:))));
%outC2 = setdiff(VV(2,:),intersect(VV(2,:),union(VV(1,:),VV(3,:))));
%outC3 = setdiff(VV(3,:),intersect(VV(3,:),union(VV(2,:),VV(1,:))));
outC1 = setdiff(VV(1,:),VV(2,:));
outC2 = setdiff(VV(2,:),VV(1,:));
% outC3 = setdiff(VV(3,:),VV(2,:));

n1 = length(outC1);
n2 = length(outC2);
% n3 = length(outC3);

extra = setdiff(Z,outC1);
extra = setdiff(extra,outC2);
% extra = setdiff(extra,outC3);

A = aa+bb;

C1 = A([outC1],extra);
d1 = sum(C1);
 
C2 = A([outC2],extra);
d2 = sum(C2);

% C3 = A([outC3],extra);
% d3 = sum(C3);

[~,idextra] = max([d1',d2'],[],2);
size(idextra)

outC1 = union(outC1,extra(find(idextra==1)));
outC2 = union(outC2,extra(find(idextra==2)));
% outC3 = union(outC3,extra(find(idextra==3)));

dZZ = 1.5*(a+b)/4;

C1 = A(outC2,outC1);
degZ = sum(C1);
bad12 = outC1(find(degZ> dZZ));
degZ = sum(C1,2);
bad21 = outC2(find(degZ> dZZ));

% C1 = A(outC3,outC1);
% degZ = sum(C1);
% bad13 = outC1(find(degZ> dZZ));
% degZ = sum(C1,2);
% bad31 = outC3(find(degZ> dZZ));

% C1 = A(outC3,outC2);
% degZ = sum(C1);
% bad23 = outC2(find(degZ> dZZ));
% degZ = sum(C1,2);
% bad32 = outC3(find(degZ> dZZ));

% outC1 = setdiff(outC1,union(bad12,bad13));
outC1 = union(outC1,bad21);
    
% outC2 = setdiff(outC2,union(bad21,bad23));
outC2 = union(outC2,bad12);

% outC3 = setdiff(outC3,union(bad31,bad32));
% outC3 = union(outC3,union(bad23,bad13));


%clustering Y
A =B;

C1 = A(outC1,Y);
d1 = sum(C1);

C1 = A(outC2,Y);
d2 = sum(C1);

% C1 = A(outC3,Y);
% d3 = sum(C1);

[~,idY] = max([d1',d2'],[],2);

outC1 = union(outC1,Y(find(idY==1)));
outC2 = union(outC2,Y(find(idY==2)));
% outC3 = union(outC3,Y(find(idY==3)));

outC1 = outC1(randperm(length(outC1)));
outC2 = outC2(randperm(length(outC2)));
% outC3 = outC3(randperm(length(outC3)));

ord = [outC1;outC2];
toc;

A = aa+bb;
spy(A(ord,ord))
figure(2)
ord2 = randperm(n);
spy(A(ord2,ord2))

end


