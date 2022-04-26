load iris.mat;
class1=Label_legend{4};
class2=Label_legend{5};
class3=Label_legend{6};

num_class_1=sum((Y_label_test==1))+sum((Y_label_train==1));
num_class_2=sum((Y_label_test==2))+sum((Y_label_train==2));
num_class_3=sum((Y_label_test==3))+sum((Y_label_train==3));

figure();
Y=[Y_label_test ; Y_label_train];
X=[X_data_test ; X_data_train];
C = categorical(Y,[1 2 3],{class1,class2,class3});
histogram(C);
title("Histogram of Class Frequency");

R = corrcoef(X);

figure();
subplot(2,3,1);
scatter(X(:,1),X(:,2));
title("Feature 2 vs. Feature 1");

subplot(2,3,2);
scatter(X(:,1),X(:, 3));
title("Feature 3 vs. Feature 1");

subplot(2,3,3);
scatter(X(:,1),X(:,4));
title("Feature 4 vs. Feature 1");

subplot(2,3,4);
scatter(X(:,2),X(:,3));
title("Feature 3 vs. Feature 2");

subplot(2,3,5);
scatter(X(:,2),X(:,4));
title("Feature 4 vs. Feature 2");

subplot(2,3,6);
scatter(X(:,3),X(:,4));
title("Feature 4 vs. Feature 3");



d=4;
m=3;
theta=zeros(d+1,m);
[y_hat_final_train, P_train,theta_final_train,G_train, CCR_train]  =SGD(X_data_train, Y_label_train,theta);
    
G_train=G_train./105; 
figure(); 
plot(G_train);
title("L-2 regularized logistic loss vs. iteration number"); 
xlabel("Iteration number");
ylabel("Logistic Loss"); 

CCR_train=CCR_train./105; 
figure(); 
plot(CCR_train);
title("CCR of the training set vs. iteration number"); 
xlabel("Iteration number");
ylabel("CCR"); 



[y_hat_final_test,P_test,theta_final_test,G_test, CCR_test]  =SGD(X_data_test, Y_label_test,theta_final_train);
CCR_test=CCR_test./45; 
figure(); 
plot(CCR_test);
title("CCR of the test set vs. iteration number"); 
xlabel("Iteration number");
ylabel("CCR"); 

figure(); 
plot(P_test);
title("Log-loss test set vs. iteration number"); 
xlabel("Iteration number");
ylabel("Log-loss"); 

C_train= confusionmat(Y_label_train,y_hat_final_train);
C_test= confusionmat(Y_label_test,y_hat_final_test);  

A= [ 1 2 3 4]; 
C = nchoosek(A,2);
Y=[Y_label_test ; Y_label_train];
X=[X_data_test ; X_data_train];
figure();
X_subset=zeros(150,4); 
 for i=1:6
     x_ind=C(i,:);
     X_subset(:,x_ind(1))=X(:,x_ind(1));
     X_subset(:,x_ind(2))=X(:,x_ind(2));
     zero_ind=setdiff( A,x_ind); 
      X_subset(:,zero_ind(1))=zeros(150,1);
     X_subset(:,zero_ind(2))=zeros(150,1);
    
     [y_hat, P,theta,G, CCR]  =SGD(X_subset, Y,theta_final_train);
     subplot(2,3,i);
     gscatter( X_subset(:,x_ind(1)), X_subset(:,x_ind(2)),y_hat);
     title("Decision Boundary for Features " + x_ind(2) + " vs " + x_ind(1));
     xlabel("Feature " + x_ind(1)); 
     ylabel("Feature " + x_ind(2));
 end 
     
     

function loss =find_reg_log_loss (theta,X,Y)
n = vecnorm(theta);
lambda=.1;
f_0= sum(n.^2)*lambda; 
f_j_sum=0; 
[r,~]=size(X); 
for i=1:r
    x=X(i,:);
    x_ext=[x';1];
    sum_log=0; 
    for l=1:3 
        sum_log=sum_log + exp (theta(:,l)'*x_ext); 
    end 
    sum_label=0;
    for l=1:3 
        sum_label=sum_label+ (l==Y(i))*theta(:,l)'*x_ext; 
    end 
    if(sum_log<10^(-10))
        sum_log=10^(-10);
    end 
   f_j_sum= f_j_sum+ log(sum_log) -sum_label;  
end 
loss=f_0 +f_j_sum;
end 
function [y_hat, ccr] = find_CCR(theta,X,Y)
[r,~]=size(X); 
y_hat=zeros(r,1); 
for i=1:r
     x=X(i,:);
    x_ext=[x';1];
    vec=theta'*x_ext; 
    [~,I]=max(vec);
    y_hat(i)=I; 
end 
ccr = sum( y_hat==Y); 
end 

function loss= find_log_loss(theta,X,Y)
 [r, ~]=size(X); 
 sum=0;
 for i=1:r 
        j=Y(i); 
        x=X(i,:);
        x_ext=[x';1];
        num=exp(theta(:,j)'*x_ext);
        denom=exp(theta(:,1)'*x_ext)+exp(theta(:,2)'*x_ext)+exp(theta(:,3)'*x_ext);
        p=num/denom;
        if(p<10^(-10))
            p=10^(-10);
        end 
        sum=sum+log(p); 
 end 
   loss=-(1/r)*sum;  

end 

function [y_hat,P,theta_final,G, CCR] = SGD (X,Y,theta)
    d=4;
m=3;
lambda=.1;
%theta=zeros(d+1,m);
v_k=zeros(d+1,m);
G=zeros(300,1); 
CCR=zeros(300,1); 
P=zeros(300,1); 
    [r, ~]=size(X); 
    for i=1:6000
    ind= randi(r);
    for j=1:3
        x=X(ind,:);
        x_ext=[x';1];
        num=exp(theta(:,j)'*x_ext);
        denom=exp(theta(:,1)'*x_ext)+exp(theta(:,2)'*x_ext)+exp(theta(:,3)'*x_ext);
        p=num/denom;
        v_k(:,j)=2*lambda*theta(:,j) + r* (p - (j==Y(ind)))*x_ext;
        if( mod(i,20)==0 && j==Y(ind))
            P(i/20)=p; 
        end 
    end
    theta=theta-((.01)/i)*v_k;
    if( mod(i,20)==0)
        G(i/20)=find_reg_log_loss(theta,X,Y); 
        [y_hat, ccr] =find_CCR(theta,X,Y); 
        CCR(i/20)=ccr; 
        P(i/20)=find_log_loss(theta,X,Y); 
    end  
    end 
      theta_final=theta; 
end 










