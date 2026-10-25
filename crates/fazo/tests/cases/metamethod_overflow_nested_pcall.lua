local function index(n)
  local t = setmetatable({}, {__index = function()
    local ok, err = pcall(index, n + 1)
    if not ok then error(err, 0) end
    return err
  end})
  return t.x
end
print(pcall(index, 0))

local function rethrow(n)
  local t = setmetatable({}, {__index = function()
    local ok, err = pcall(rethrow, n + 1)
    error({depth = n, inner = err}, 0)
  end})
  return t.x
end
local ok, err = pcall(rethrow, 0)
local depth = 0
while type(err) == "table" do depth = depth + 1; err = err.inner end
print(ok, depth > 50, err)

local function handled(n)
  local t = setmetatable({}, {__index = function()
    return select(2, xpcall(handled, function(m) return "h:" .. tostring(m) end, n + 1))
  end})
  return t.x
end
print(pcall(handled, 0))

local function deep(n)
  if n == 0 then
    local t = setmetatable({}, {__index = function() return deep(0) end})
    return t.x
  end
  return pcall(deep, n - 1)
end
local results = table.pack(deep(150))
print(results.n, results[1], results[results.n - 1], results[results.n])

local function eq(n)
  local mt = {__eq = function() local ok, e = pcall(eq, n + 1) error(e, 0) end}
  return setmetatable({}, mt) == setmetatable({}, mt)
end
print(pcall(eq, 0))

local function concat(n)
  local t = setmetatable({}, {__concat = function() error(select(2, pcall(concat, n + 1)), 0) end})
  return t .. "x"
end
print(pcall(concat, 0))
