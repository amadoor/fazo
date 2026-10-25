local log = {}
local function closer(name)
  return setmetatable({}, {__close = function(_, err)
    log[#log + 1] = name .. ":" .. tostring(err)
  end})
end
do
  local a <close> = closer("a")
  local b <close> = closer("b")
  log[#log + 1] = "body"
end
print(table.concat(log, " "))
log = {}
print(pcall(function()
  local c <close> = closer("c")
  error("boom", 0)
end))
print(table.concat(log, " "))
log = {}
local function early()
  local d <close> = closer("d")
  return "returned"
end
print(early(), table.concat(log, " "))
print(pcall(function() local x <close> = 42 end))
local nothing <close> = nil
print("nil is closable", nothing)
