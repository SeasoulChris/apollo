import flask_restful


class AccountService(flask_restful.Resource):
    def get(self):
        return {"hello": "world"}

    def post(self):
        return {"msg": "success"}
