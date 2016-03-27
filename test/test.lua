local fg = require 'factorgraph'

local TestFG = torch.TestSuite()
local tester = torch.Tester()

local var_a = function()
  return fg.Variable(3, 'a')
end

local var_b = function()
  return fg.Variable(2, 'b')
end

local factor_b = function(b)
  return fg.Factor(torch.Tensor{0.3, 0.7}, {b}, 'P_B')
end

local factor_a_given_b = function(a, b)
  return fg.Factor(torch.Tensor{{0.2, 0.8}, {0.4, 0.6}, {0.1, 0.9}}, {a, b}, 'P_A_given_B')
end

function TestFG.test_variable_init()
  local a = var_a()
  tester:asserteq('a', a.name)
  tester:asserteq(1e-5, a.convergence)
  tester:asserteq(3, a.n_states)
  tester:asserteq(nil, a.observed_state)
end

function TestFG:test_factor_init()
  local b = var_b()
  local P_b = factor_b(b)

  tester:asserteq('P_B', P_b.name)
  tester:asserteq(1e-5, P_b.convergence)
  tester:assertTensorEq(torch.Tensor{0.3, 0.7}, P_b.factor.P, 1e-5)
  tester:assertTensorEq(torch.ones(2), P_b.incoming[b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), P_b.outgoing[b].P, 1e-5)

  tester:assertTensorEq(torch.ones(2), b.outgoing[P_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.outgoing[P_b].P, 1e-5)
end

function TestFG:test_forward()
  local a = var_a()
  local b = var_b()
  local P_b = factor_b(b)
  local P_a_given_b = factor_a_given_b(a, b)
  P_b:forward()
  P_a_given_b:forward()
  tester:assertTensorEq(torch.Tensor{0.3, 0.7}, P_b.outgoing[b].P, 1e-5)
  tester:assertTensorEq(torch.Tensor{0.7/3, 2.3/3}, P_a_given_b.outgoing[b].P, 1e-5)
  tester:assertTensorEq(torch.Tensor{1/3, 1/3, 1/3}, P_a_given_b.outgoing[a].P, 1e-5)

  tester:assertTensorEq(torch.ones(3), a.incoming[P_a_given_b].P)
  tester:assertTensorEq(torch.ones(2), b.incoming[P_a_given_b].P)
  tester:assertTensorEq(torch.ones(2), b.incoming[P_b].P)
  P_b:send()
  tester:assertTensorEq(torch.ones(2), b.incoming[P_a_given_b].P)
  tester:assertTensorEq(torch.Tensor{0.3, 0.7}, b.incoming[P_b].P)

  P_a_given_b:send()
  tester:assertTensorEq(torch.Tensor{1/3, 1/3, 1/3}, a.incoming[P_a_given_b].P, 1e-5)
  tester:assertTensorEq(torch.Tensor{0.7/3, 2.3/3}, b.incoming[P_a_given_b].P, 1e-5)
end

function TestFG:test_believe_propagation()
  local a = var_a()
  local b = var_b()
  local P_b = factor_b(b)
  local P_a_given_b = factor_a_given_b(a, b)
  local g = fg.Graph({a, b}, {P_a_given_b, P_b})
  g:forward{max_iter=10}
  local m = g:marginals()
  tester:assertTensorEq(torch.Tensor{0.11538, 0.88462}, m[b].P, 1e-5)
  tester:assertTensorEq(torch.Tensor{0.34066, 0.29670, 0.36264}, m[a].P, 1e-5)

  g:reset()
  tester:assertTableEq({}, a.prev_outgoing)
  tester:assertTableEq({}, b.prev_outgoing)
  tester:assertTableEq({}, P_b.prev_outgoing)
  tester:assertTableEq({}, P_b.prev_outgoing)
  tester:assertTensorEq(torch.ones(3), a.incoming[P_a_given_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(3), a.outgoing[P_a_given_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.incoming[P_a_given_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.outgoing[P_a_given_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.incoming[P_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.outgoing[P_b].P, 1e-5)

  tester:assertTensorEq(torch.ones(2), P_b.outgoing[b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), P_a_given_b.outgoing[b].P, 1e-5)
  tester:assertTensorEq(torch.ones(3), P_a_given_b.outgoing[a].P, 1e-5)

  g:forward{max_iter=10}
  tester:assertTensorEq(torch.Tensor{0.11538, 0.88462}, m[b].P, 1e-5)
  tester:assertTensorEq(torch.Tensor{0.34066, 0.29670, 0.36264}, m[a].P, 1e-5)
end

function TestFG:test_reset()
  local a = var_a()
  a:condition(1)
  a.incoming[1] = tl.ProbTable(torch.Tensor{1, 2, 3}, 'a')
  a.outgoing[1] = tl.ProbTable(torch.Tensor{3, 2, 1}, 'a')
  a:reset()
  tester:asserteq(nil, a.observed_state)
  tester:assertTensorEq(torch.ones(3), a.incoming[1].P, 1e-5)
  tester:assertTensorEq(torch.ones(3), a.outgoing[1].P, 1e-5)

  local b = var_b()
  local P_b = factor_b(b)
  P_b:reset()
  tester:assertTensorEq(torch.ones(2), P_b.incoming[b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), P_b.outgoing[b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.incoming[P_b].P, 1e-5)
  tester:assertTensorEq(torch.ones(2), b.outgoing[P_b].P, 1e-5)
end

function TestFG:test_condition()
  local b = var_b()
  b.outgoing[1] = tl.ProbTable(torch.Tensor{1, 2}, 'b')
  b:condition(2)
  tester:assertTensorEq(torch.Tensor{0, 1}, b.outgoing[1].P, 1e-5)
end

function TestFG:marginal()
  local b = var_b()
  b.incoming[1] = tl.ProbTable(torch.Tensor{1, 2}, 'b')
  b.incoming[2] = tl.ProbTable(torch.Tensor{3, 4}, 'b')
  local joint = b:marginal()
  tester:assertTensorEq(torch.Tensor{3, 8}, joint.P, 1e-5)
end

tester:add(TestFG)
tester:run()
