local function closer(name)
  return setmetatable({}, {__close = function(_, msg)
    print("close", name, msg)
    coroutine.yield(name)
    print("resumed", name)
  end})
end
local co = coroutine.create(function()
  print(pcall(function()
    local a <close> = closer("a")
    local b <close> = closer("b")
    error("boom", 0)
  end))
  return "done"
end)
for _ = 1, 4 do print(coroutine.resume(co)) end
