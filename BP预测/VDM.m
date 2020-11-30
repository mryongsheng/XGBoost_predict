function    attribute=VDM(data,label,ClassType,AttVector)
if(sum(unique(AttVector)==[0,1])~=2)
    error('AttVector error')
end

NumClass=length(ClassType);
NumAtt=length(AttVector); 
for i=1:NumAtt
    if(AttVector(i)==0)   
        attribute(i).kind='numeric';
        attribute(i).values=[];       
        attribute(i).VDM=[];
    else
        attribute(i).kind='nominal';
        attribute(i).values=unique(data(i,:)); 
        N=length(attribute(i).values);
        n=zeros(1,N);
        for k=1:N
            n(k)=length(find(abs(data(i,:)-attribute(i).values(k))<1e-6));
        end
        
        attribute(i).VDM=zeros(N);
        for ui=1:N
            for vi=ui+1:N
                if(vi~=ui)               
                    u=attribute(i).values(ui);
                    v=attribute(i).values(vi);
                    Nu=n(ui);
                    Nv=n(vi);
                    d=0;
                    for j=1:NumClass
                        Nuc=length( intersect( find(data(i,:)==u), find(label==ClassType(j)) ) );
                        Nvc=length( intersect( find(data(i,:)==v), find(label==ClassType(j)) ) );
                        d=d+((Nuc/Nu)-(Nvc/Nv))^2;
                    end  
                    attribute(i).VDM(ui,vi)=d;  
                    attribute(i).VDM(vi,ui)=d;  
                end       
            end
        end
        
    end%if-else
end
