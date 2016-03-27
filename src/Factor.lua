--[[ Factor node implementation. ]]
local Factor, Parent = torch.class('fg.Factor', 'fg.Node')
local Table = require('torchlib').ProbTable

--[[ Constructor. ]]
function Factor:__init(prob, connections, name, convergence)
  assert(#connections > 0, 'connections must be non-empty')
  assert(#connections == prob:nDimension(), 'number of connections must equal number of dimensions of prob')
  Parent.__init(self, name, convergence)

  -- initialize probability table
  local names = {}
  for i, c in ipairs(connections) do
    table.insert(names, c.name)
    assert(prob:size(i) == c.n_states, 'prob['..i..' has '..prob:size(i)..' dimensions but variable '..c.name..' has '..c.n_states..' states')
  end
  self.factor = Table(prob, names)

  -- initialize messages
  for i, variable in ipairs(connections) do
    self.incoming[variable] = variable:uniform_message()
    self.outgoing[variable] = variable:uniform_message()

    variable.incoming[self] = variable:uniform_message()
    variable.outgoing[self] = variable:uniform_message()
  end
end

--[[ Resets the node. ]]
function Factor:reset()
  Parent.reset(self)
  for variable, message in pairs(self.incoming) do
    self.incoming[variable] = variable:uniform_message()
  end
  for variable, message in pairs(self.outgoing) do
    self.outgoing[variable] = variable:uniform_message()
    self.prev_outgoing[variable] = nil
  end
end

--[[ Combine messages by summing and multiplying. ]]
function Factor:forward()
  if self.enabled and self.observed_state == nil then
    self:next()
    for o, out_message in pairs(self.outgoing) do
      out_message = self.factor
      for i, in_message in pairs(self.incoming) do
        if i ~= o then
          out_message = out_message:mul(in_message)
        end
      end
      out_message:marginal(o.name)
      self.outgoing[o] = out_message
    end
  end
  self:normalize()
end

return Factor
