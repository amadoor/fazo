local function make(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
  return { a = a, b = b, c = c, d = d, e = e, f = f, g = g, h = h, i = i, j = j, k = k, l = l, m = m, n = n, o = o, p = p }
end

local function escapes_on_one_path(a, b, c, d, e, f, g, h, keep)
  local record = { a = a, b = b, c = c, d = d, e = e, f = f, g = g, h = h }
  if keep then
    return record
  end
  return record.a
end

local built, kept
for count = 1, 60000 do
  local mixed = (count % 2 == 0) and count or tostring(count)
  local half = count * 0.5
  built = make(mixed, half, mixed, half, mixed, half, mixed, half, mixed, half, mixed, half, mixed, half, mixed, half)
  kept = escapes_on_one_path(mixed, half, mixed, half, mixed, half, mixed, half, count % 1000 == 0) or kept
end

assert(built.a == 60000 and built.p == 30000.0 and built.h == 30000.0 and built.o == 60000)
assert(type(kept) == "table" or kept == 60000)
print(built.a, built.b, built.o, built.p)
