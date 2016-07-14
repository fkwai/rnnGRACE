
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

--require('mobdebug').start()

folder="/Volumes/wrgroup/Kuai/rnnGRACE/"



tabGRACE=csv2tensor(folder .. "tabGRACE.csv")
tabUSGS=csv2tensor(folder .. "tabUSGS.csv")
ID=csv2tensor(folder .. "tabID.csv")
timeGRACE=csv2tensor(folder .. "timeGRACE.csv")
timeUSGS=csv2tensor(folder .. "timeUSGS.csv")


batchSize = 50
hiddenSize = 64
inputSize = 1
outputSize = 1
lr=0.01
niter=1000

model = nn.Sequential()
model:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))

criterion = nn.SequencerCriterion(nn.MSECriterion())


for i = 1, 1 do
    input, target = {}, {}
    sel = torch.LongTensor(batchSize):random(1,batchSize)
    for j=1,timeUSGS:size(1) do
        input[j]=torch.Tensor()
        input[j]:index(tabUSGS[j],1,sel)
        input[j]=input[j]:view(batchSize,1)
    end
    for j=1,timeGRACE:size(1) do
        target[j]=torch.Tensor()
        target[j]:index(tabGRACE[j],1,sel)
        target[j]=target[j]:view(batchSize,1)
    end

    model:zeroGradParameters()

    output = model:forward(input)

    output_t={}
    for j=1,timeGRACE:size(1) do
    	output_t[j]=torch.Tensor()
    	ind=torch.range(1,timeUSGS:nElement())[timeUSGS:eq(timeGRACE[j])]
    	output_t[j]=output[ind[1]]
    end
    
    err = criterion:forward(output_t, target)

    print('error for iteration ' .. i  .. ' is ' .. err/timeGRACE:size(1))
  
    gradOutput_t = criterion:backward(output_t, target)

    input_t={}
    for j=1,timeUSGS:size(1) do
    	ind=torch.range(1,timeGRACE:nElement())[timeGRACE:eq(timeUSGS[j])]
    	if ind:dim()~=0 then
    		input_t[j]=input[ind[1]]
    	end
    end

    model:backward(input, gradOutput)
    model:updateParameters(lr)


end



