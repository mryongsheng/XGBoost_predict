
function [sample,sampleLabel]=SmoteOverSampling(data,Label,ClassType,C,AttVector,k,type)

%check parameters
NumClass=size(ClassType,2);
if(size(C)~=NumClass)
    error('class number and cost matrix do not consistent.')
end
if(size(data,2)~=size(Label))
    error('instance numbers in data and Label do not consistent.')
end
if(size(data,1)~=length(AttVector))
    error('attribute numbers in data and AttVector do not consistent.')
end

%prepare for distance function
attribute=VDM(data,Label,ClassType,AttVector);

%compute class distribution
ClassD=zeros(1,NumClass);
for i=1:NumClass
    id=find(Label==ClassType(i));
    ClassData{i}=data(:,id);
    ClassD(i)=length(id);
end
%compute new class distribution
cn=C./ClassD;
[tmp,baseClassID]=min(cn);
newClassD=floor(C/C(baseClassID)*ClassD(baseClassID));

%over sampling using SMOTE
sample=data;
sampleLabel=Label;
clear data Label;
i=1;
while(i<NumClass | i==NumClass )    
    if(newClassD(i)>ClassD(i))
        diff=newClassD(i)-ClassD(i);       
        s=SMOTE(ClassData{1},diff,k,type,attribute,AttVector);    
        sample=[sample s];
        sampleLabel=[sampleLabel repmat(ClassType(i),1,diff)];         
    end
    if(length(ClassData)>1)
        ClassData=ClassData(2:end);% delete after used
    end
    i=i+1;
end
end