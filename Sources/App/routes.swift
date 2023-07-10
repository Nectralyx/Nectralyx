import Vapor

func routes(_ app: Application) throws {
    app.get { req async in
        "It works!"
    }

    app.get("hello") { req async -> String in
        "Hello, world!"
    }
    app.post("send", use: echo)
    app.get("test") { req async -> String in
"""
---OverDrive DataBase---
        
           /|
          /||
         //||
--------///||*******///-
-------///*||******///--
------///**||*****///---
-----///***||****///----
----///****||***///-----
---///*****||**///------
--///******||*///-------
-///*******||///--------
           ||//
           |//
           |/
               
               
               
"""
        
    }
    func echo(request: Request) async throws -> String {
            if let body = request.body.string {
                return body
            }
            return ""
        }
    
    
}

