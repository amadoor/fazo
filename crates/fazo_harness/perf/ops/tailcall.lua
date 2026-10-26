local x = 1; local function g(a) return a end; local function f(a) return g(a) end
for i = 1, 200000 do
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
end
print(x)
