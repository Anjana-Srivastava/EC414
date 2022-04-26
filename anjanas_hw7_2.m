iris=load('iris.mat');
x_test=iris.X_data_test;
x2_test=x_test(:,2);
x4_test=x_test(:,4);
x_test=[x2_test x4_test];
y_test=iris.Y_label_test;

x_train=iris.X_data_train;
x2_train=x_train(:,2);
x4_train=x_train(:,4);
x_train=[x2_train x4_train];
y_train=iris.Y_label_train;

%1 Binary classifier class 1 is +1 class 2 is -1
yind= (y_train==1 | y_train==2 );
y_class=y_train(yind);
y_class(y_class==2)=-1;
x_class=x_train(yind,:);
[theta_1, cost_1,ccr_1]=SSGD(x_class,y_class);
[n,~]=size(y_class);
cost_1=cost_1./n;
ccr_1=ccr_1./n;
figure();
subplot(3,1,1);
plot(cost_1);
title("Cost vs. iteration number for classifying class 1 and 2");
xlabel("Iteration number");
ylabel("Cost");

y_hat_train_1=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_train_1(i)=sign (theta_1'*x_ext');
end

C_train_1= confusionmat(y_hat_train_1,y_class); 

%2 Binary classifier class 1 is +1 class 3 is -1
yind= (y_train==1 | y_train==3 );
y_class=y_train(yind);
y_class(y_class==3)=-1;
x_class=x_train(yind,:);
[theta_2, cost_2,ccr_2]=SSGD(x_class,y_class);
[n,~]=size(y_class);
ccr_2=ccr_2./n;
cost_2=cost_2./n;
subplot(3,1,2);
plot(cost_2);
title("Cost vs. iteration number for classifying class 1 and 3");
xlabel("Iteration number");
ylabel("Cost");


y_hat_train_2=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_train_2(i)=sign (theta_2'*x_ext');
end

C_train_2= confusionmat(y_hat_train_2,y_class); 


%3 Binary classifier class 2 is +1 class 3 is -1
yind= (y_train==2 | y_train==3 );
y_class=y_train(yind);
y_class(y_class==2)=1;
y_class(y_class==3)=-1;
x_class=x_train(yind,:);
[theta_3, cost_3,ccr_3]=SSGD(x_class,y_class);
[n,~]=size(y_class);
cost_3=cost_3./n;
ccr_3=ccr_3./n;
subplot(3,1,3);
plot(cost_3);
title("Cost vs. iteration number for classifying class 2 and 3");
xlabel("Iteration number");
ylabel("Cost");


y_hat_train_3=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_train_3(i)=sign (theta_3'*x_ext');
end

C_train_3= confusionmat(y_hat_train_3,y_class); 


%plot training ccrs
figure();
subplot(3,1,1);
plot(ccr_1);
title("Training ccr  for classifying class 1 and 2");
xlabel("Iteration number");
ylabel("CCR");

subplot(3,1,2);
plot(ccr_2);
title("Training ccr  for classifying class 1 and 3");
xlabel("Iteration number");
ylabel("CCR");

subplot(3,1,3);
plot(ccr_3);
title("Training ccr  for classifying class 2 and 3");
xlabel("Iteration number");
ylabel("CCR");


%plot test ccr
%class 1 vs class 2 (-1) 
yind= (y_test==1 | y_test==2 );
y_class=y_test(yind);
y_class(y_class==2)=-1;
x_class=x_test(yind,:);
[theta_1_test, cost_1_test,ccr_1_test]=SSGD(x_class,y_class);
[n,~]=size(y_class);
cost_1_test=cost_1_test./n;
ccr_1_test=ccr_1_test./n;
figure();
subplot(3,1,1);
plot(ccr_1_test);
title("Testing ccr  for classifying class 1 and 2");
xlabel("Iteration number");
ylabel("CCR");


y_hat_test_1=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_test_1(i)=sign (theta_1_test'*x_ext');
end

C_test_1= confusionmat(y_hat_test_1,y_class); 

%class 1 vs class 3 (-1) 
yind= (y_test==1 | y_test==3 );
y_class=y_test(yind);
y_class(y_class==3)=-1;
x_class=x_test(yind,:);
[theta_2_test, cost_2_test,ccr_2_test]=SSGD(x_class,y_class);
[n,~]=size(y_class);
cost_2_test=cost_2_test./n;
ccr_2_test=ccr_2_test./n;

subplot(3,1,2);
plot(ccr_2_test);
title("Testing ccr  for classifying class 1 and 3");
xlabel("Iteration number");
ylabel("CCR");

y_hat_test_2=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_test_2(i)=sign (theta_2_test'*x_ext');
end

C_test_2= confusionmat(y_hat_test_2,y_class); 

%class 2 (1) vs. class 3 (-1) 
yind= (y_test==2 | y_test==3 );
y_class=y_test(yind);
y_class(y_class==2)=1;
y_class(y_class==3)=-1;
x_class=x_test(yind,:);
[theta_3_test, cost_3_test,ccr_3_test]=SSGD(x_class,y_class);
[n,~]=size(y_class);
cost_3_test=cost_3_test./n;
ccr_3_test=ccr_3_test./n;
subplot(3,1,3);
plot(ccr_3_test);
title("Testing ccr  for classifying class 2 and 3");
xlabel("Iteration number");
ylabel("CCR")


y_hat_test_3=zeros(n,1);
for i=1:n
    x_ext=[x_class(i,:) 1];
    y_hat_test_3(i)=sign (theta_3_test'*x_ext');
end

C_test_3= confusionmat(y_hat_test_3,y_class); 

%all pairs train 
[n,~]=size(x_train); 
final_pred_train=zeros(n,1); 
for i=1:n 
    count=zeros(1,3);
    x_ext=[x_train(i,:) 1]; 
    
    theta_1_pred=sign (theta_1'*x_ext');
    if(theta_1_pred==1)
        count(1)=count(1)+1;
    else
         count(2)=count(2)+1;
    end 
    
      theta_2_pred=sign (theta_2'*x_ext');
    if(theta_2_pred==1)
        count(1)=count(1)+1;
    else
         count(3)=count(3)+1;
    end
    
      theta_3_pred=sign (theta_3'*x_ext');
    if(theta_3_pred==1)
        count(2)=count(2)+1;
    else
         count(3)=count(3)+1;
    end 
    
    [~,ind] = max(count); 
    final_pred_train(i)=ind; 
end 

ccr_all_pairs_train=sum(final_pred_train==y_train)/n; 
c_all_pairs_train=confusionmat(y_train,final_pred_train);

%all pairs test 
[n,~]=size(x_test);
count=zeros(1,3); 
final_pred_test=zeros(n,1); 
for i=1:n 
    count=zeros(1,3);
    x_ext=[x_test(i,:) 1]; 
    
    theta_1_pred=sign (theta_1_test'*x_ext');
    if(theta_1_pred==1)
        count(1)=count(1)+1;
    else
         count(2)=count(2)+1;
    end 
    
      theta_2_pred=sign (theta_2_test'*x_ext');
    if(theta_2_pred==1)
        count(1)=count(1)+1;
    else
         count(3)=count(3)+1;
    end
    
      theta_3_pred=sign (theta_3_test'*x_ext');
    if(theta_3_pred==1)
        count(2)=count(2)+1;
    else
         count(3)=count(3)+1;
    end 
    
    [~,ind] = max(count); 
    final_pred_test(i)=ind; 
end 

ccr_all_pairs_test=sum(final_pred_test==y_test)/n; 
c_all_pairs_test=confusionmat(y_test,final_pred_test);

function [theta,cost,ccr] =SSGD(x,y)
d=2;
theta=zeros(d+1,1);
t_max=2*10^5;
s=.5;
C=1.2;
[n, d] = size(x);
I = eye(d);
zc = zeros(d,1);
I=[I zc];
zc = zeros(d+1,1);
I=[I; zc'];
cost=zeros(200,1);
ccr=zeros(200,1);
for i=1:t_max
    ind= randi(n);
    v=I*theta;
    x_ext=[x(ind,:) 1];
    if(y(ind)*theta'*x_ext'<1)
        v=v-n*C*y(ind)*x_ext';
    end
    theta=theta-(s/i)*v;
    
    if(i==1) 
        ccr(i)=find_ccr(theta,x,y);    
         cost(i)=find_cost(theta,x,y);
    elseif( mod(i,1000)==0)
           ccr(i/1000 +1)=find_ccr(theta,x,y);
            cost(i/1000)=find_cost(theta,x,y);
        end
          
    end
end



function [cost] = find_cost(theta,x,y)
[n,~]=size(theta);
w=theta(1:n-1,1);
b=theta(n,1);

f_0=.5*norm(w)^2;
[n,~]=size(x);
sum=0;
for i=1:n
    x_ext=[x(i,:) 1];
    val=1-y(i)*theta'*x_ext';
    if(val>0)
        sum=sum+1.2*val;
    end
end
cost=f_0+sum;
end

function [ccr]= find_ccr(theta,x,y)
[n,~]=size(x);
y_hat=zeros(n,1);
for i=1:n
    x_ext=[x(i,:) 1];
    y_hat(i)=sign (theta'*x_ext');
end
ccr= sum(y==y_hat);

end