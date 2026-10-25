local function closer(name)
  return setmetatable({}, {
    __close = function(_, err)
      print("close", name, err)
    end,
  })
end
local ok, err = pcall(function()
  local a <close> = closer("a")
  local b <close> = closer("b")
  error("boom", 0)
end)
print(ok, err)
do
  local c <close> = closer("c")
  print("body")
end
