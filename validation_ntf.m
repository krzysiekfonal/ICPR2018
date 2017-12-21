function out = validation_ntf(UC_orig,UA_orig,UB_orig,UC,UA,UB,res,elapsed_time)

if ~isempty(UC_orig{1})
    
    N = length(UC_orig);
    for n = 1:N
        SIRC(n,:) = CalcSIR(UC_orig{n},UC{n});
        SIRA(n,:) = CalcSIR(UA_orig{n},UA{n});
        SIRB(n,:) = CalcSIR(UB_orig{n},UB{n});
    end
    
figure
subplot(2,2,1)
bar(SIRC)
title(['Mean-SIR = ',num2str(round(mean2(SIRC))),', ET = ',num2str(elapsed_time),' [sec.]']);
ylabel('SIR C [dB]','FontName','Times New Roman','FontSize',18)
xlabel('Modes','FontName','Times New Roman','FontSize',18)
set(gca,'FontName','Times New Roman','FontSize',20)

subplot(2,2,2)
bar(SIRA)
title(['Mean-SIR = ',num2str(round(mean2(SIRA))),', ET = ',num2str(elapsed_time),' [sec.]']);
ylabel('SIR A [dB]','FontName','Times New Roman','FontSize',18)
xlabel('Modes','FontName','Times New Roman','FontSize',18)
set(gca,'FontName','Times New Roman','FontSize',20)

subplot(2,2,3)
bar(SIRB)
title(['Mean-SIR = ',num2str(round(mean2(SIRB))),', ET = ',num2str(elapsed_time),' [sec.]']);
ylabel('SIR B [dB]','FontName','Times New Roman','FontSize',18)
xlabel('Modes','FontName','Times New Roman','FontSize',18)
set(gca,'FontName','Times New Roman','FontSize',20)

subplot(2,2,4)
semilogy(res)
grid on
title('Residuals');
ylabel('Normalized residuals','FontName','Times New Roman','FontSize',18)
xlabel('Iterations','FontName','Times New Roman','FontSize',18)
set(gcf,'Color',[1 1 1])
set(gca,'FontName','Times New Roman','FontSize',20)
    
else
    
figure
plot(res)
grid on
title(['ET = ',num2str(elapsed_time),' [sec.]']);
ylabel('Normalized residuals','FontName','Times New Roman','FontSize',18)
xlabel('Iterations','FontName','Times New Roman','FontSize',18)
set(gcf,'Color',[1 1 1])
set(gca,'FontName','Times New Roman','FontSize',20)
    
end



