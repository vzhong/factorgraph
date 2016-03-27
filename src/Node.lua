--[[ Abstract Node class. ]]
local Node = torch.class('fg.Node')

--[[ Automatic identifier. ]]
function Node:register()
  local tab = torch.getmetatable(torch.type(self))
  tab.num_exist = (tab.num_exist or 0) + 1
end

function Node:get_global_count()
  local tab = torch.getmetatable(torch.type(self))
  return tab.num_exist or 0
end

--[[ Constructor. ]]
function Node:__init(name, convergence)
  self:register()
  self.name = name or torch.type(self)..self:get_global_count()
  self.incoming = {}
  self.outgoing = {}
  self.prev_outgoing = {}
  self.convergence = convergence or 1e-5
  self.enabled = true
end

--[[ Resets and empties the incoming and outgoing buffers. ]]
function Node:reset()
  self.enabled = true
end

--[[ Receives a message from a node. ]]
function Node:receive(node, message)
  if self.enabled then
    self.incoming[node] = message
  end
end

--[[ Sends a message to all listeners. ]]
function Node:send()
  for node, message in pairs(self.outgoing) do
    node:receive(self, message)
  end
end

--[[ Prepare for new iteration by caching outgoing messages. ]]
function Node:next()
  for node, message in pairs(self.outgoing) do
    self.prev_outgoing[node] = message:clone()
  end
end

--[[ Normalizes the messages in the outgoing buffer to sum to 1. ]]
function Node:normalize()
  for node, message in pairs(self.outgoing) do
    message:normalize()
  end
end

--[[ Returns if outgoing message did not change. ]]
function Node:converged()
  if not self.enabled then return true end
  for node, message in pairs(self.outgoing) do
    local prev_message = self.prev_outgoing[node]
    if prev_message == nil or (torch.csub(message.P, prev_message.P):abs():max() > self.convergence) then
      return false
    end
  end
  return true
end

return Node
