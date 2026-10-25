local Point = {}
Point.__index = Point
function Point.new(x, y) return setmetatable({x = x, y = y}, Point) end
function Point:add(other) return Point.new(self.x + other.x, self.y + other.y) end
function Point:len2() return self.x * self.x + self.y * self.y end
local p = Point.new(0, 0)
local d = Point.new(1, 2)
local s = 0
for i = 1, 150000 do
    p = p:add(d)
    s = s + p:len2() % 97
end
print(p.x, p.y, s)
