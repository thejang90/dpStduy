import numpy as np
import unittest

class Variable:
    '''
    변수를 담을 클래스 생성
    '''
    def __init__(self,data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None # 미분값을 저장하는 변수
        self.creator = None # 해당 변수를 생성한 함수 저장
        self.generation = 0 # 세대수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 부모 세대 보다 한 세대 더 늘어남

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None :
                    x.grad = gx
                else :
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple) :
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs # 입력 변수를 보관, backward시 필요함
        self.outputs = outputs # 출력 변수 저장
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def add(x0, x1) :
    return Add()(x0,x1)

def square(x):
    return Square()(x)

def exp(x):
    return  Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


## Test 부분
# x = Variable(np.array(2.0))
# y = Variable(np.array(3.0))
# x2 = Variable(np.array(4.0))
#
# a = square(x)
# y = add(square(a), square(a))
# y.backward()
#
# print(y.data)
# print(x.grad)

generations = [2,0,1,4,2]
func =[]

for g in generations :
    f = Function()
    f.generation = g
    func.append(f)

func.sort(key=lambda x: x.generation)
print([f.generation for f in func])