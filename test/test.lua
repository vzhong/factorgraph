local fg = require 'factorgraph'

local b = fg.Variable(2, 'B')
local a = fg.Variable(3, 'A')

local P_b = fg.Factor(torch.Tensor{0.3, 0.7}, {b}, 'P_B')

local P_a_given_b = fg.Factor(torch.Tensor{{0.2, 0.8}, {0.4, 0.6}, {0.1, 0.9}}, {a, b}, 'P_A_given_B')

local g = fg.Graph({a, b}, {P_a_given_b, P_b})


--P_b:forward()
--P_b:send()
--
--P_a_given_b:forward()
--P_a_given_b:send()
--
--print(P_b.outgoing[b])
--print(b.incoming[P_b])
--
--b:forward()
--b:send()
--print(b.outgoing[P_a_given_b])
--
--print(P_a_given_b.outgoing[a])

print('converged in '..g:forward({max_iter=10})..' iterations')
for node, p in pairs(g:marginals()) do
  print(p)
end


