--[[ Factor graph implementation. ]]
local Graph = torch.class('fg.Graph')
local tl = require 'torchlib'

--[[ Constructor. ]]
function Graph:__init(variables, factors)
  self.variables = variables or {}
  self.factors = factors or {}
  self.all = tl.util.tableCopy(self.variables)
  self.all = tl.util.extend(self.all, self.factors)
  self.converged = false
end

--[[ Returns the total number of nodes in the graph. ]]
function Graph:size()
  return #self.all
end

--[[ Believe propagation algorithm. ]]
function Graph:forward(opt)
  opt = opt or {}
  opt.max_iter = opt.max_iter or 50
  for t = 1, opt.max_iter do
    if self.converged then return t end
    local order = {}
    for _, i in ipairs(torch.randperm(#self.all):totable()) do
      table.insert(order, self.all[i])
    end
    -- propagate
    for _, node in ipairs(order) do
      node:forward()
      node:send()
    end

    -- check convergence
    self.converged = true
    for _, node in ipairs(order) do
      self.converged = self.converged and node:converged()
      if not self.converged then break end
    end
  end
end

--[[ Compute the marginals after BP. ]]
function Graph:marginals()
  local marginals = {}
  for _, var in ipairs(self.variables) do
    if var.enabled then
      marginals[var] = var:marginal():normalize()
    end
  end
  return marginals
end

--[[ Reset all nodes. ]]
function Graph:reset()
  for _, node in ipairs(self.all) do node:reset() end
end

return Graph
