local function nest(n)
  if n == 0 then return 0 end
  return coroutine.wrap(function() return nest(n - 1) + 1 end)()
end
local ok, err = pcall(nest, 300)
print(ok, select(2, err:gsub(":%d+: ", "")))
