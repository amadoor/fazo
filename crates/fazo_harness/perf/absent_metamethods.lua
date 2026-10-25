local plain = {}
local bare = setmetatable({}, {})
local sum = 0
for i = 1, 300000 do
    if plain.missing == nil then sum = sum + 1 end
    if bare.missing == nil then sum = sum + 2 end
    if rawget(bare, "x") == nil then sum = sum + 3 end
    if getmetatable(bare).__newindex == nil then sum = sum + 4 end
end
print(sum)
