
require "rnn"
require "nn"
require "csvigo"


function csv2tensor(filename)
	tab=csvigo.load({path=filename,mode="large"})

	local nrow=#tab
	local ncol=#tab[1]
	if ncol==1 then
		data = torch.Tensor(nrow)
		for j=1,nrow do
			next_col=torch.Tensor(tab[j])
			data[{{j}}]:copy(next_col)
		end
	else
		data = torch.Tensor(nrow, ncol)
	end
	
	for j=1,nrow do
		next_col=torch.Tensor(tab[j])
		data[{{j}}]:copy(next_col)
	end
	return data
end


folder="/Volumes/wrgroup/Kuai/rnnGRACE/"

inputField={}
inputField[1]="Storage"
inputField[2]="Rainf"
inputField[3]="Snowf"
inputField[4]="SWnet"
inputField[5]="LWnet"
inputField[6]="Qair"
inputField[7]="Wind"

targetField={}
targetField[1]="GRACE"

tabInput={}
for i=1,#inputField do
    tabInput[i]=torch.Tensor()
    tab=csv2tensor(folder .. "gridTab".. inputField[i]  .."_norm.csv")
    tabInput[i]=tab
end

tabTarget={}
for i=1,#targetField do
    tabTarget[i]=torch.Tensor()
    tab=csv2tensor(folder .. "gridTab".. inputField[i]  .."_norm.csv")
    tabTarget[i]=tab
end

--nstep=tabInput[1]:size(2) 
nstep=96
nindv=tabInput[1]:size(1)
inputSize = #inputField
outputSize = #targetField

-- argument of rnn
batchSize = 100
hiddenSize = 100
lr=0.01
niter=5000
rho=12


model = nn.Sequential()
model:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))

criterion = nn.SequencerCriterion(nn.MSECriterion())

-- training

for iter=1,niter do
	indsel,stepsel={},{}
    for ibatch=1,batchSize do
    	indsel[ibatch]=math.random(1,nindv)
    	stepsel[ibatch]=math.random(1,nstep-rho+1)
    end

    input,target={},{}
    for istep=1,rho do
        input[istep]=torch.Tensor(batchSize,inputSize):zero()
        for ivar=1,inputSize do
            tab=torch.Tensor(batchSize):zero()
		    for ibatch=1,batchSize do
		    	temp=tabInput[ivar]
		    	tab[ibatch]=temp[indsel[ibatch]][stepsel[ibatch]+istep-1]
		    end 
            --tab=tabInput[ivar]:index(1,indsel):select(2,istep)
            input[istep]:select(2,ivar):copy(tab)
        end          
        target[istep]=torch.Tensor(batchSize,outputSize):zero()
        for ivar=1,outputSize do
            tab=torch.Tensor(batchSize):zero()
		    for ibatch=1,batchSize do
		    	temp=tabTarget[ivar]
		    	tab[ibatch]=temp[indsel[ibatch]][stepsel[ibatch]+istep-1]
		    end
            --tab=tabTarget[ivar]:index(1,indsel):select(2,istep)
            target[istep]:select(2,ivar):copy(tab)
        end 
    end
    model:zeroGradParameters()

    output = model:forward(input)
    err = criterion:forward(output, target)

    if iter % 10 == 0 then
        print('error for iteration ' .. iter  .. ' is ' .. err/rho)
    end

    gradOutput = criterion:backward(output, target)
    model:backward(input, gradOutput)
    model:updateParameters(lr)
end

--]]

-- output
x={}
ind1=1
ind2=1000
step1=1
step2=96
ni=ind2-ind1+1
ns=step2-step1+1
for istep=1,ns do

	if istep % 10 == 0 then
        print('loading time step:' .. istep)
    end

    x[istep]=torch.Tensor(ni,inputSize):zero()
    for ivar=1,inputSize do
        tab=torch.Tensor(ni):zero()
	    for ibatch=1,ni do
	    	temp=tabInput[ivar]
	    	tab=temp[{{ind1,ind2},{step1+istep-1}}]
	    end 
        --tab=tabInput[ivar]:index(1,indsel):select(2,istep)
        x[istep]:select(2,ivar):copy(tab)
    end          
end

print("predicting")
y=model:forward(x)
print("done")

out={}
for j=1,#y do
	if j % 10 == 0 then
        print('converting grid:' .. j)
    end
	col=y[j]
	out[j]={}
	for i=1,y[1]:size(1) do
		out[j][i]=col[i][1]
	end
end
csvigo.save({path=folder .. "out",data=out,separator=","})

outrange={}
outrange[1]={}
outrange[1][1]=ind1
outrange[1][2]=ind2
outrange[2]={}
outrange[2][1]=step1
outrange[2][2]=step2
csvigo.save({path=folder .. "out_range",data=outrange,separator=","})



--[[
out = assert(io.open(folder .. "/out.csv", "w"))
indout=1
ni=y[1]:size(1)
for j=1,#y do

	if j % 10 == 0 then
        print('writing grid:' .. j)
    end

	tab=y[j]
	for i=1,ni-1 do
		out:write(tab[i][indout])
		out:write(",")
	end
	out:write(tab[ni][indout])
	out:write("\n")
end
out:close()
--]]

