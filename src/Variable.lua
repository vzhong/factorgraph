--[[ Variable node implementation. ]]
local Variable, Parent = torch.class('fg.Variable', 'fg.Node')
local Table = require('torchlib').ProbTable

--[[ Constructor. ]]
function Variable:__init(n_states, name, convergence)
  Parent.__init(self, name, convergence)
  self.n_states = n_states
  self.observed_state = nil
end

--[[ Returns a probability table initialized to 1 of a size equal to this variable.
By default, `fill` is 1.
--]]
function Variable:uniform_message(fill)
  return Table(torch.Tensor(self.n_states):fill(fill or 1), self.name)
end

--[[ Resets the node. ]]
function Variable:reset()
  Parent.reset(self)
  self.observed_state = nil
  for node, message in pairs(self.incoming) do
    message = self:uniform_message()
  end
  for node, message in pairs(self.outgoing) do
    message = self:uniform_message()
    self.prev_outgoing[node] = nil
  end
end

--[[ Sets the node to a conditioned value. ]]
function Variable:condition(state)
  self.observed_state = state
  for node, message in pairs(self.outgoing) do
    message.P:zero()
    message.P[self.observed_state] = 1
  end
end

--[[ Combine messages by multiplying. ]]
function Variable:forward()
  if self.enabled and self.observed_state == nil then
    self:next()
    for o, out_message in pairs(self.outgoing) do
      out_message = self:uniform_message()
      for i, in_message in pairs(self.incoming) do
        if i ~= o then
          out_message = out_message:mul(in_message)
        end
      end
      self.outgoing[o] = out_message
    end
  end
  self:normalize()
end

--[[ Returns the marginal of this variable by combining incoming messages. ]]
function Variable:marginal()
  if not self.enabled then return nil end
  local joint
  for node, message in pairs(self.incoming) do
    if joint then joint = joint:mul(message) else joint = message:clone() end
  end
  return joint
end

return Variable
